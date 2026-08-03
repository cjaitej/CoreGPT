"""
Modern nanoGPT: GPT-2 with Llama-inspired architectural improvements.

Same training recipe as train_gpt2.py -- the architecture is flag-gated instead,
so this one file runs every row of the comparison table:

    # exact GPT-2 baseline (parity check against train_gpt2.py)
    python train_modern_gpt.py --pos=learned --norm=layernorm

    # + RoPE
    python train_modern_gpt.py --pos=rope --norm=layernorm

    # + RoPE + RMSNorm  ("modern")
    python train_modern_gpt.py --pos=rope --norm=rmsnorm

The KV cache is inference-only and does not affect training; see benchmark.py.

DDP launch (e.g. Kaggle's 2x T4):
    torchrun --standalone --nproc_per_node=2 train_modern_gpt.py
"""

import os
import math
import time
import json
import inspect
import argparse
from dataclasses import dataclass, asdict

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# Normalization

class RMSNorm(nn.Module):
    """Root-mean-square layer norm (Zhang & Sennrich 2019), as used by Llama.

    LayerNorm subtracts the mean and divides by the standard deviation, then
    applies a learned scale and shift. RMSNorm drops the mean-centering and the
    bias entirely:

        y = x / sqrt(mean(x^2) + eps) * weight

    The claim is that re-centering contributes little to LayerNorm's benefit --
    the re-scaling does the work -- so dropping it saves a pass over the feature
    dimension and one parameter tensor per norm with no loss in quality.

    The statistics are computed in fp32 even under autocast: the mean of squares
    of a 768-wide bf16 vector loses meaningful precision otherwise.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_norm(config, dim):
    """Norm factory. Attribute names stay ln_* so GPT-2 checkpoints still load."""
    if config.norm == "rmsnorm":
        return RMSNorm(dim)
    elif config.norm == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm: {config.norm}")

# -----------------------------------------------------------------------------
# Rotary position embeddings (RoPE)

def rotate_half(x):
    """(..., d) -> (..., d), pairing dim i with dim i + d/2 and rotating."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary(q, k, cos, sin):
    """Rotate q and k in place of adding a position embedding.

    q, k:      (B, nh, T, hs)
    cos, sin:  (1, 1, T, hs), fp32

    Done in fp32 then cast back -- the rotation is a similarity-preserving
    transform and doing it in bf16 injects avoidable error into every score.
    """
    dtype = q.dtype
    q, k = q.float(), k.float()
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out.to(dtype), k_out.to(dtype)


class RotaryEmbedding(nn.Module):
    """Precomputed cos/sin tables for RoPE (Su et al. 2021).

    Learned absolute position embeddings add a vector to the token embedding
    once, at the bottom of the network, and the model has to carry that
    information up through every layer. RoPE instead rotates q and k by an
    angle proportional to their absolute position, at every layer. Because the
    attention score is an inner product,

        <R(m) q, R(n) k> = <q, R(n - m) k>

    the score depends only on the *relative* offset n - m. Position information
    is injected where it is actually used, and it never occupies a slot in the
    residual stream.

    Tables are registered non-persistently: they are a deterministic function of
    (head_dim, max_seq_len, theta), so keeping them out of the state dict means
    checkpoints stay portable across context-length changes.
    """

    def __init__(self, head_dim, max_seq_len, theta=10000.0):
        super().__init__()
        assert head_dim % 2 == 0, "RoPE needs an even head dimension"
        # inv_freq[i] = 1 / theta^(2i/d): low dims rotate fast (local detail),
        # high dims rotate slowly (long-range position).
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                 # (T, hs/2)
        emb = torch.cat((freqs, freqs), dim=-1)          # (T, hs), matches rotate_half's pairing
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :], persistent=False)

    def forward(self, pos_start, seq_len):
        """Slice the tables for absolute positions [pos_start, pos_start+seq_len).

        pos_start is non-zero when decoding with a KV cache: token number 500
        must be rotated by its true position, not by 0 just because it is the
        only token in this forward pass.
        """
        end = pos_start + seq_len
        assert end <= self.cos_cached.size(2), (
            f"position {end} exceeds RoPE table length {self.cos_cached.size(2)}")
        return self.cos_cached[:, :, pos_start:end], self.sin_cached[:, :, pos_start:end]

# -----------------------------------------------------------------------------
# KV cache

class KVCache:
    """Preallocated per-layer key/value cache for autoregressive decoding.

    Without a cache, generating token t re-runs the whole prefix through every
    layer: total work over n generated tokens is O(n^2) forward passes worth of
    projections. But keys and values for a given position never change once
    computed -- causal masking means position i cannot see anything after it.
    So we compute them once and keep them.

    Buffers are allocated up front rather than grown by torch.cat, which would
    reallocate and copy the entire cache on every single token.

    Memory cost: 2 * n_layer * B * max_seq_len * n_head * head_dim * itemsize.
    """

    def __init__(self, n_layer, batch_size, max_seq_len, n_head, head_dim, device, dtype):
        shape = (batch_size, n_head, max_seq_len, head_dim)
        self.k = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.v = [torch.zeros(shape, device=device, dtype=dtype) for _ in range(n_layer)]
        self.max_seq_len = max_seq_len
        self.pos = 0  # number of valid cached tokens

    def update(self, layer_idx, k, v):
        """Append this step's k/v for one layer, return everything cached so far.

        Writes at self.pos for every layer; advance() is called once per forward
        pass after all layers have written, so they stay aligned.
        """
        T = k.size(2)
        assert self.pos + T <= self.max_seq_len, "KV cache overflow"
        self.k[layer_idx][:, :, self.pos:self.pos + T] = k
        self.v[layer_idx][:, :, self.pos:self.pos + T] = v
        return self.k[layer_idx][:, :, :self.pos + T], self.v[layer_idx][:, :, :self.pos + T]

    def advance(self, n):
        self.pos += n

    def reset(self):
        self.pos = 0

    def memory_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.k + self.v)

# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x, rope=None, kv_cache=None, layer_idx=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # RoPE is applied to q and k only -- v carries content, not position
        if rope is not None:
            cos, sin = rope
            q, k = apply_rotary(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(layer_idx, k, v)

        # is_causal aligns its mask to the top-left of the (q_len, kv_len) score
        # matrix. During cached decode q_len == 1 while kv_len == pos + 1, so
        # is_causal=True would let the new token see only position 0. Every
        # cached position is a valid target for a single new query, so no mask
        # is needed there. Chunked prefill (T > 1 into a non-empty cache) would
        # need a real mask and is not supported.
        assert not (T > 1 and kv_cache is not None and kv_cache.pos > 0), \
            "chunked prefill is not supported"
        y = F.scaled_dot_product_attention(q, k, v, is_causal=(T > 1))
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = build_norm(config, config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = build_norm(config, config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x, rope=None, kv_cache=None, layer_idx=None):
        x = x + self.attn(self.ln_1(x), rope, kv_cache, layer_idx)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # number of tokens: 50,000 BPE merges + 256 bytes tokens + 1 <|endoftext|> token
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding dimension
    pos: str = "rope" # "learned" (GPT-2) or "rope"
    norm: str = "rmsnorm" # "layernorm" (GPT-2) or "rmsnorm"
    rope_theta: float = 10000.0

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        if config.pos not in ("learned", "rope"):
            raise ValueError(f"unknown pos: {config.pos}")
        # Submodules are created in the same order as train_gpt2.py so that a
        # given seed produces bit-identical weights in the --pos=learned
        # --norm=layernorm configuration. That makes the baseline row of the
        # comparison table a real control rather than a re-roll of the dice.
        modules = dict(wte = nn.Embedding(config.vocab_size, config.n_embd))
        if config.pos == "learned":
            modules["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        modules["h"] = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        modules["ln_f"] = build_norm(config, config.n_embd)
        self.transformer = nn.ModuleDict(modules)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.pos == "rope":
            # RoPE replaces wpe outright: block_size * n_embd fewer parameters
            # (786,432 at GPT-2 124M) and no learned table to run off the end
            # of. It holds only buffers, so it consumes no init randomness.
            self.rope = RotaryEmbedding(
                config.n_embd // config.n_head, config.block_size, config.rope_theta)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, kv_cache=None, last_only=False):
        # idx is of shape (B, T)
        B, T = idx.size()
        # with a cache, T is the number of *new* tokens; the prefix is already stored
        pos_start = kv_cache.pos if kv_cache is not None else 0
        assert pos_start + T <= self.config.block_size, (
            f"Cannot forward sequence of length {pos_start + T}, "
            f"block size is only {self.config.block_size}")

        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        rope = None
        if self.config.pos == "learned":
            pos = torch.arange(pos_start, pos_start + T, dtype=torch.long, device=idx.device)
            x = tok_emb + self.transformer.wpe(pos)
        else:
            rope = self.rope(pos_start, T)
            x = tok_emb

        # forward the blocks of the transformer
        for i, block in enumerate(self.transformer.h):
            x = block(x, rope, kv_cache, i)
        if kv_cache is not None:
            kv_cache.advance(T)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        # Sampling only ever reads the last position, but lm_head is the single most
        # expensive op in a small model (~65% of FLOPs at n_embd=384 against a 50304
        # vocab). Projecting the whole prefill and discarding all but the last row
        # measured 63.9ms vs 6.2ms on CPU at a 91-token prompt.
        if last_only and targets is None:
            x = x[:, -1:]
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def new_kv_cache(self, batch_size, max_seq_len=None, device=None, dtype=torch.float32):
        cfg = self.config
        return KVCache(
            n_layer=cfg.n_layer,
            batch_size=batch_size,
            max_seq_len=max_seq_len or cfg.block_size,
            n_head=cfg.n_head,
            head_dim=cfg.n_embd // cfg.n_head,
            device=device or next(self.parameters()).device,
            dtype=dtype,
        )

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=50,
                 use_cache=True, generator=None, kv_dtype=torch.float32):
        """Sample max_new_tokens continuations of idx (B, T).

        use_cache=False reproduces the original nanoGPT loop: the entire
        sequence is re-forwarded for every token. use_cache=True forwards the
        prompt once, then one token at a time.
        """
        self.eval()
        B = idx.size(0)
        cache = None
        if use_cache:
            total = idx.size(1) + max_new_tokens
            assert total <= self.config.block_size, (
                f"{total} tokens exceeds block size {self.config.block_size}")
            cache = self.new_kv_cache(B, total, idx.device, kv_dtype)

        for step in range(max_new_tokens):
            if cache is None:
                # no cache: feed the whole (cropped) sequence every time
                idx_cond = idx[:, -self.config.block_size:]
                logits, _ = self(idx_cond, last_only=True)
            else:
                # cache: the prompt on step 0, then only the newest token
                idx_cond = idx if step == 0 else idx[:, -1:]
                logits, _ = self(idx_cond, kv_cache=cache, last_only=True)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            if top_k is not None:
                topk_probs, topk_indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=generator)
                next_token = torch.gather(topk_indices, -1, ix)
            else:
                next_token = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat((idx, next_token), dim=1)
        return idx

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # the released checkpoints are learned-position + LayerNorm by construction
        config_args['pos'] = 'learned'
        config_args['norm'] = 'layernorm'
        config_args.update(override_args or {})
        assert config_args['pos'] == 'learned' and config_args['norm'] == 'layernorm', (
            "OpenAI GPT-2 weights only load into --pos=learned --norm=layernorm; "
            "RoPE and RMSNorm variants must be trained from scratch")
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type, verbose=True):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        # (RMSNorm's gain is 1D, so it lands in the no-decay group like LayerNorm's.)
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if verbose:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if verbose:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -----------------------------------------------------------------------------

def pick_amp_dtype(device_type):
    """bf16 only where there is hardware for it, otherwise fp16.

    Deliberately NOT torch.cuda.is_bf16_supported(): that defaults to
    including_emulation=True, so on Turing (SM 7.5, e.g. Kaggle's T4) the
    compute-capability test fails but it falls through to an emulation check
    that succeeds and returns True. Turing has fp16 tensor cores and no bf16
    ones, so taking it at its word puts every matmul on a non-tensor-core path.
    Measured on a Kaggle T4: ~11k tok/s under emulated bf16, where the point of
    this function was to select fp16 in exactly that case.

    bf16 tensor cores start at SM 8.0 (Ampere).
    """
    if device_type != "cuda":
        return torch.bfloat16
    major, _ = torch.cuda.get_device_capability()
    return torch.bfloat16 if major >= 8 else torch.float16


def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split, data_dir, verbose=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        if os.path.isdir(data_dir):
            # get the shard filenames
            shards = sorted(s for s in os.listdir(data_dir) if split in s)
            shards = [os.path.join(data_dir, s) for s in shards]
            self.shards = shards
            assert len(shards) > 0, f"no shards found for split {split} in {data_dir}"
            if verbose:
                print(f"found {len(shards)} shards for split {split}")
        elif os.path.isfile(data_dir):
            # single raw text file (e.g. input.txt) -- for local smoke tests.
            # 90/10 split, tokenized once and held in memory.
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            with open(data_dir, "r", encoding="utf-8") as f:
                tokens = enc.encode(f.read())
            n = int(0.9 * len(tokens))
            tokens = tokens[:n] if split == "train" else tokens[n:]
            self.shards = None
            self._tokens = torch.tensor(tokens, dtype=torch.long)
            if verbose:
                print(f"loaded {len(tokens):,} {split} tokens from {data_dir}")
            assert len(tokens) > B * T * num_processes + 1, \
                f"{data_dir} has too few {split} tokens for B={B} T={T}"
        else:
            raise FileNotFoundError(
                f"--data_dir {data_dir!r} is neither a shard directory nor a file. "
                f"Run fineweb.py to build it, or pass --data_dir=input.txt")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[0]) if self.shards else self._tokens
        self.current_position = self.B * self.T * self.process_rank

    def state_dict(self):
        # store the position with this rank's stride removed, so only rank 0
        # needs to checkpoint and every rank still resumes to its own offset
        return {"current_shard": self.current_shard,
                "current_position": self.current_position - self.B * self.T * self.process_rank}

    def load_state_dict(self, state):
        self.current_shard = state["current_shard"]
        self.tokens = load_tokens(self.shards[self.current_shard]) if self.shards else self._tokens
        self.current_position = state["current_position"] + self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            if self.shards:
                self.current_shard = (self.current_shard + 1) % len(self.shards)
                self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y

# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # architecture ablation switches
    p.add_argument("--pos", choices=["learned", "rope"], default="rope")
    p.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="rmsnorm")
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # model size. Defaults are a ~30M "small" model that finishes an ablation
    # inside one Kaggle session; see README for the 124M GPT-2 flags.
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--vocab_size", type=int, default=50304) # padded to a multiple of 128
    # optimization
    p.add_argument("--batch_size", type=int, default=16, help="micro batch size B")
    p.add_argument("--seq_len", type=int, default=512, help="sequence length T")
    p.add_argument("--total_batch_size", type=int, default=131072, help="tokens per optimizer step")
    p.add_argument("--max_steps", type=int, default=8000)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--max_lr", type=float, default=6e-4)
    p.add_argument("--min_lr_frac", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # data / io
    p.add_argument("--data_dir", default="edu_fineweb10B",
                   help="shard directory, or a raw .txt file for smoke tests")
    p.add_argument("--out_dir", default="log")
    p.add_argument("--run_name", default=None, help="defaults to <pos>_<norm>")
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--val_steps", type=int, default=20)
    p.add_argument("--ckpt_every", type=int, default=1000,
                   help="Kaggle sessions are capped at ~12h; checkpoint often")
    p.add_argument("--resume", action="store_true", help="resume from ckpt.pt in the run dir")
    p.add_argument("--hellaswag", action="store_true", help="run HellaSwag eval (slow)")
    p.add_argument("--sample_every", type=int, default=0, help="0 disables sampling during training")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto",
                   help="auto picks bf16 where supported, else fp16; override to test the "
                        "fp16 path on a bf16-capable card before running on Kaggle's T4")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main():
    args = get_args()

    # -------------------------------------------------------------------------
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        from torch.distributed import init_process_group
        assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        ddp_rank, ddp_local_rank, ddp_world_size = 0, 0, 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        print(f"using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"

    # Kaggle's T4 is SM 7.5 and has no bf16 support, so pick the dtype from the
    # hardware instead of hardcoding it. fp16 has the same 10-bit mantissa as
    # bf16 but far less exponent range, so it needs loss scaling to keep small
    # gradients from flushing to zero; bf16 does not.
    amp_dtype = getattr(torch, args.dtype) if args.dtype != "auto" else pick_amp_dtype(device_type)
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
    if master_process:
        cap = f"SM {'.'.join(map(str, torch.cuda.get_device_capability()))}" \
              if device_type == "cuda" else device_type
        print(f"autocast dtype: {amp_dtype} on {cap} (GradScaler: {use_scaler})")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')

    # -------------------------------------------------------------------------
    B, T = args.batch_size, args.seq_len
    assert args.total_batch_size % (B * T * ddp_world_size) == 0, \
        "make sure total_batch_size is divisible by B * T * ddp_world_size"
    grad_accum_steps = args.total_batch_size // (B * T * ddp_world_size)
    if master_process:
        print(f"total desired batch size: {args.total_batch_size}")
        print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    train_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, "train", args.data_dir, master_process)
    val_loader = DataLoaderLite(B, T, ddp_rank, ddp_world_size, "val", args.data_dir, master_process)

    # create model
    config = GPTConfig(
        block_size=args.block_size, vocab_size=args.vocab_size,
        n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
        pos=args.pos, norm=args.norm, rope_theta=args.rope_theta,
    )
    model = GPT(config)
    model.to(device)
    if master_process:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"config: {config}")
        print(f"parameters: {n_params:,}")

    raw_model = model
    if args.compile:
        model = torch.compile(model)
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module

    min_lr = args.max_lr * args.min_lr_frac
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < args.warmup_steps:
            return args.max_lr * (it + 1) / args.warmup_steps
        # 2) if it > lr_decay_iters, return min learning rate
        if it > args.max_steps:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_steps) / (args.max_steps - args.warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        return min_lr + coeff * (args.max_lr - min_lr)

    optimizer = raw_model.configure_optimizers(
        args.weight_decay, args.max_lr, device_type, verbose=master_process)

    # -------------------------------------------------------------------------
    # run directory, logging, resume
    run_name = args.run_name or f"{args.pos}_{args.norm}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file = os.path.join(run_dir, "log.txt")
    ckpt_path = os.path.join(run_dir, "ckpt.pt")

    start_step = 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if use_scaler and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        train_loader.load_state_dict(ckpt["train_loader"])
        start_step = ckpt["step"] + 1
        if master_process:
            print(f"resumed from {ckpt_path} at step {start_step}")
    elif master_process:
        with open(log_file, "w") as f:
            pass
        with open(os.path.join(run_dir, "config.json"), "w") as f:
            json.dump({**asdict(config), **vars(args)}, f, indent=2)

    def save_checkpoint(step, val_loss):
        torch.save({
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict() if use_scaler else None,
            'train_loader': train_loader.state_dict(),
            # asdict, not the dataclass instance: pickling GPTConfig records it as
            # __main__.GPTConfig, which fails to load from any other entry point
            # (benchmark.py, a notebook) where __main__ is a different module
            'config': asdict(config),
            'args': vars(args),
            'step': step,
            'val_loss': val_loss,
        }, ckpt_path)

    def log(line):
        if master_process:
            with open(log_file, "a") as f:
                f.write(line + "\n")

    # -------------------------------------------------------------------------
    import tiktoken
    import torch.distributed as dist
    enc = tiktoken.get_encoding("gpt2")
    last_val_loss = float("nan")

    for step in range(start_step, args.max_steps):
        t0 = time.time()
        last_step = (step == args.max_steps - 1)

        # once in a while evaluate our validation loss
        if step % args.eval_every == 0 or last_step:
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss_accum = 0.0
                for _ in range(args.val_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    with torch.autocast(device_type=device_type, dtype=amp_dtype):
                        logits, loss = model(x, y)
                    val_loss_accum += (loss / args.val_steps).detach()
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            last_val_loss = val_loss_accum.item()
            if master_process:
                # perplexity = exp(cross-entropy); the headline number for the report
                print(f"validation loss: {last_val_loss:.4f} | ppl: {math.exp(last_val_loss):.2f}")
                log(f"{step} val {last_val_loss:.4f}")

        # once in a while evaluate hellaswag
        if args.hellaswag and (step % args.eval_every == 0 or last_step) and not args.compile:
            from hellaswag import render_example, iterate_examples
            num_correct_norm = 0
            num_total = 0
            for i, example in enumerate(iterate_examples("val")):
                # only process examples where i % ddp_world_size == ddp_rank
                if i % ddp_world_size != ddp_rank:
                    continue
                _, tokens, mask, label = render_example(example)
                tokens, mask = tokens.to(device), mask.to(device)
                with torch.no_grad():
                    with torch.autocast(device_type=device_type, dtype=amp_dtype):
                        logits, _ = model(tokens)
                    pred_norm = get_most_likely_row(tokens, mask, logits)
                num_total += 1
                num_correct_norm += int(pred_norm == label)
            if ddp:
                stats = torch.tensor([num_total, num_correct_norm], dtype=torch.long, device=device)
                dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                num_total, num_correct_norm = stats.tolist()
            acc_norm = num_correct_norm / num_total
            if master_process:
                print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
                log(f"{step} hella {acc_norm:.4f}")

        # once in a while generate from the model (except step 0, which is noise)
        if args.sample_every and step > 0 and step % args.sample_every == 0 and not args.compile:
            model.eval()
            tokens = enc.encode("Hello, I'm a language model,")
            xgen = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0).repeat(4, 1)
            sample_rng = torch.Generator(device=device)
            sample_rng.manual_seed(42 + ddp_rank)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                xgen = raw_model.generate(xgen, 32 - xgen.size(1), top_k=50,
                                          use_cache=True, generator=sample_rng)
            for i in range(xgen.size(0)):
                print(f"rank {ddp_rank} sample {i}: {enc.decode(xgen[i].tolist())}")

        # do one step of the optimization
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            # added after video, this field is also used by the forward pass.
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            with torch.autocast(device_type=device_type, dtype=amp_dtype):
                logits, loss = model(x, y)
            # we have to scale the loss to account for gradient accumulation,
            # because the gradients just add on each successive backward().
            # addition of gradients corresponds to a SUM in the objective, but
            # instead of a SUM we want MEAN. Scale the loss here so it comes out right
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            scaler.scale(loss).backward()
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        # unscale before clipping so grad_clip means the same thing in fp16 and bf16
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        scaler.step(optimizer)
        scaler.update()
        if device_type == "cuda":
            torch.cuda.synchronize() # wait for the GPU to finish work
        dt = time.time() - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | "
                  f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
            log(f"{step} train {loss_accum.item():.6f}")

        # Checkpoint at the *end* of the iteration, so ckpt["step"] == N means
        # step N is complete and resuming at N+1 neither repeats nor drops an
        # optimizer update. Kaggle sessions die at ~12h; save often.
        if master_process and (step % args.ckpt_every == 0 or last_step):
            save_checkpoint(step, last_val_loss)

    if ddp:
        from torch.distributed import destroy_process_group
        destroy_process_group()


if __name__ == "__main__":
    main()

# Modern nanoGPT

Modernizing GPT-2 with Llama-inspired architectural improvements and efficient inference.

GPT-2 (2019) and Llama (2023) are both decoder-only transformers, and the diff between
them is smaller than it looks. This repo starts from a from-scratch GPT-2 reproduction
([`train_gpt2.py`](train_gpt2.py), following [karpathy/build-nanogpt](https://github.com/karpathy/build-nanogpt))
and changes three things, one at a time, measuring each:

| Feature | Purpose | Retrain? | Where |
| :--- | :--- | :---: | :--- |
| **RoPE** | Relative positional encoding | yes | `RotaryEmbedding`, `apply_rotary` |
| **RMSNorm** | Simpler, cheaper normalization | yes | `RMSNorm` |
| **KV cache** | O(n²) → O(n) decoding | no | `KVCache` |

`train_gpt2.py` is untouched and serves as the control.

```
train_gpt2.py          original GPT-2 reproduction, unmodified
train_modern_gpt.py    flag-gated model + training loop  <- the project
benchmark.py           inference / training / correctness harness
kaggle_train.ipynb     dual-T4 notebook: two ablations at once
fineweb.py             data prep (now supports a --shards cap)
```

The architecture is selected by flags rather than by editing the model, so one file
produces every row of the comparison table and the baseline is a genuine control:

```bash
python train_modern_gpt.py --pos=learned --norm=layernorm   # GPT-2 baseline
python train_modern_gpt.py --pos=rope    --norm=layernorm   # + RoPE
python train_modern_gpt.py --pos=rope    --norm=rmsnorm     # + RMSNorm ("modern")
```

## Correctness first

An architecture comparison is worthless if the baseline is secretly a different model, or
if the "optimization" quietly changes the outputs. Three properties are asserted in code:

```bash
python benchmark.py --mode=check
```

```
--- baseline parity (train_gpt2.py vs --pos=learned --norm=layernorm) ---
29 tensors, max weight diff 0.0e+00, max logit diff 0.0e+00
PASS: baseline is bit-identical to train_gpt2.py

--- RoPE relative position property ---
offset +4 at absolute 3/10/50: -5.480939, -5.480939, -5.480941 (spread 1.91e-06)
offset +6 for contrast: -9.231455
PASS: RoPE scores are translation invariant

--- KV cache equivalence ---
max |no-cache - cached| logit difference: 2.027e-06 (logit scale 2.294)
PASS: KV cache is numerically equivalent
```

The parity check matters most. Submodules in `train_modern_gpt.py` are constructed in the
same order as the original, so under the same seed `--pos=learned --norm=layernorm`
produces *bit-identical weights*. Any loss gap in the table below is therefore
attributable to the architecture, not to a different roll of the RNG.

---

## 1. RoPE — Rotary Position Embeddings

**Problem.** GPT-2 learns a `(block_size, n_embd)` table and adds row `i` to token `i`'s
embedding, once, at the bottom of the network. Three consequences: position information
must survive 12 layers of residual updates while competing for room in the same vector
space as content; the model learns *absolute* positions, so a phrase at position 900 looks
different from the same phrase at position 5; and the table has a hard edge — there is no
row 1025, so context cannot be extended without training new parameters.

**Change.** Delete `wpe`. Instead, rotate the query and key vectors of *every* layer by an
angle proportional to their absolute position.

```
GPT-2                             RoPE
-----                             ----
tok_emb + pos_emb                 tok_emb
      |                                 |
   Block x N                        Block x N
                                        |  Q,K -> rotate by position -> Q',K'
                                        |  V untouched (it carries content)
                                     Attention
```

**Intuition.** Split each head's dimensions into pairs and treat each pair as a point in a
2D plane. Rotate pair *j* of the vector at position *m* by angle *m·θⱼ*. Because a rotation
is orthogonal, the attention inner product becomes

```
<R(m)q, R(n)k> = qᵀ R(m)ᵀ R(n) k = qᵀ R(n - m) k
```

The score depends only on the **relative** offset `n - m`. Position is injected exactly
where it is used — inside the attention score — and never occupies a slot in the residual
stream. The frequencies `θⱼ = 10000^(-2j/d)` span fast rotations in early dimensions
(local detail) through very slow ones in later dimensions (long-range order): the same
construction as sinusoidal encodings, applied multiplicatively instead of additively.

That identity is what the check verifies numerically: the score spread across absolute
positions 3, 10 and 50 at a constant offset is 1.9e-06, i.e. zero up to fp32 noise.

**Trade-offs.** Rotation costs two elementwise multiplies and a `rotate_half` on Q and K in
every layer, versus one embedding lookup for the whole forward pass — a small but real
throughput cost, paid to remove `block_size × n_embd` parameters (786,432 at GPT-2 124M).
It is applied in fp32 and cast back, since performing a similarity-preserving rotation in
bf16 injects avoidable error into every attention score. RoPE also does not extrapolate for
free beyond its trained context; that needs YaRN/NTK scaling (future work).

## 2. RMSNorm

**Problem.** LayerNorm computes a mean and a variance, subtracts, divides, then applies a
learned scale *and* a learned shift. That is two reduction passes over the feature
dimension and two parameter tensors, executed 25 times per forward pass at GPT-2 124M.

**Change.**

```
LayerNorm:  y = (x - mean(x)) / sqrt(var(x) + eps) * weight + bias
RMSNorm:    y =  x            / sqrt(mean(x²) + eps) * weight
```

**Intuition.** The empirical finding (Zhang & Sennrich, 2019) is that LayerNorm's benefit
comes from *re-scaling*, not *re-centering* — the network is largely invariant to the mean
being carried along, because the very next operation is a linear layer that can absorb it.
Drop the mean subtraction and the bias, and one reduction pass and one parameter tensor per
norm disappear with no measured loss in quality. Llama, Mistral and Gemma all use it.

Statistics are computed in fp32 even under autocast: the mean of squares across a 768-wide
bf16 vector loses meaningful precision, and this is a normalization layer, where that error
compounds with depth.

**Trade-offs.** Fewer parameters (25 × `n_embd` = 19,200 at 124M) and a cheaper kernel, but
outputs are no longer zero-centered — visible in the checks as a surviving per-token mean.
Weight decay must still exclude the 1D gain; `configure_optimizers` handles that
automatically, since it groups by `p.dim() >= 2`.

## 3. KV cache

**Problem.** Naive generation re-runs the entire prefix through every layer for each new
token. Generating *n* tokens costs O(n²) projections, and the work is almost entirely
redundant: causal masking means position *i* cannot see anything after it, so the keys and
values at position *i* are **frozen** the moment they are computed.

```
without cache                     with cache
-------------                     ----------
prompt -> forward all             prompt -> forward once, store K,V
+1 tok -> forward all again       +1 tok -> forward 1 token, read cached K,V
+1 tok -> forward all again       +1 tok -> forward 1 token, append to cache
```

**Change.** `KVCache` preallocates `(B, n_head, max_seq_len, head_dim)` buffers per layer
and writes each step's K/V in place. Preallocation rather than `torch.cat` is deliberate:
concatenation reallocates and copies the whole cache on every single token, which reverses
much of the saving.

Two subtleties the implementation has to get right:

- **Positions.** With a cache, a forward pass sees `T=1`, but that token is not at position
  0. RoPE tables are sliced from `cache.pos`, and learned positions index from `cache.pos`,
  so token 500 is encoded as token 500.
- **Masking.** `F.scaled_dot_product_attention(is_causal=True)` aligns its triangular mask
  to the top-left of the `(q_len, kv_len)` score matrix. During decode `q_len=1` while
  `kv_len=pos+1`, so `is_causal=True` would let the new token attend *only to position 0* —
  a silent, plausible-looking corruption. Every cached position is a valid target for a
  single new query, so no mask is needed there at all.

**Trade-offs.** Speed is bought with memory that grows linearly in context and batch:
`2 × n_layer × B × T × n_head × head_dim × itemsize` (~38 MB at batch 8 and 512 tokens for
the 30M model, measured below). This is exactly what GQA and MQA later attack.

---

## Benchmarks

### Inference — KV cache

30M params (6L / 6H / 384d), batch 8, bf16, RTX 3050 Laptop (4GB):

| Prompt | Gen | KV cache | Tok/s | ms/token | Peak VRAM (MB) | Cache (MB) | Speedup |
| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 128 | no | 529.9 | 15.10 | 355.1 | – | 1.00x |
| 64 | 128 | yes | 815.8 | 9.81 | 262.1 | 14.2 | **1.54x** |
| 128 | 256 | no | 278.8 | 28.69 | 516.6 | – | 1.00x |
| 128 | 256 | yes | 775.1 | 10.32 | 328.7 | 28.3 | **2.78x** |
| 256 | 256 | no | 266.8 | 29.98 | 622.5 | – | 1.00x |
| 256 | 256 | yes | 756.2 | 10.58 | 447.1 | 37.7 | **2.83x** |

Two things worth reading off this table:

1. **The speedup grows with sequence length** — 1.54x → 2.83x — which is the O(n²) → O(n)
   argument showing up in wall-clock. Uncached `ms/token` doubles from 15.1 to 30.0 as
   context grows, while the cached figure stays near 10. Flat per-token latency regardless
   of how much context precedes it is the actual product property.
2. **Peak VRAM is *lower* with the cache**, not higher, despite the cache itself costing
   38 MB. The uncached path re-materializes activations for the entire sequence on every
   step; the cached path only ever forwards one token. The cache trades a large transient
   allocation for a smaller persistent one.

**When it does not pay.** On a 6.8M model (2 layers, 128d) generating 64 tokens from a
32-token prompt, the same benchmark measures **0.93x** — the cache is *slower*. With a
prefix that short there is barely any quadratic work to remove, and the win is swamped by
per-step kernel-launch and Python overhead, which the cached path pays once per token
instead of amortizing over a batched forward. The cache is a fix for long contexts; it is
not free at small scale.

Reproduce with `python benchmark.py --mode=inference`.

### Training throughput

| Arch | Params | B × T | ms/step | Tok/s | Peak VRAM (MB) |
| :--- | ---: | :---: | ---: | ---: | ---: |
| rope + rmsnorm | 29,959,296 | 4×512 | 87.0 | 23,538 | 1,919.9 |

Reproduce with `python benchmark.py --mode=training`. Run it once per `--pos`/`--norm`
combination to isolate each change's throughput cost.

### Quality — not yet measured

These require the three training runs below. Fill in from each run's `log/<run>/log.txt`;
perplexity is `exp(val_loss)` and is printed at every eval.

| Model | RoPE | Norm | Params | Val loss | PPL | HellaSwag |
| :--- | :---: | :--- | ---: | ---: | ---: | ---: |
| GPT-2 baseline | ✗ | LayerNorm | | | | |
| + RoPE | ✓ | LayerNorm | | | | |
| Modern | ✓ | RMSNorm | | | | |

---

## Running it

### Data

`--shards N` switches `fineweb.py` to a streaming download and stops after N shards, so
only the slice you ask for is fetched. Shard 0 is validation, the rest are training; each
is 100M tokens / 200MB of `uint16`.

```bash
python fineweb.py --shards 7      # 1 val + 6 train, ~700M tokens, ~1.4GB
python fineweb.py                 # the full 10B sample (~20GB, needs ~50GB to build)
```

Six training shards is not arbitrary: at `--max_steps=4000 --total_batch_size=131072` a run
consumes ~524M tokens, so anything less means looping over the same data twice. Scale
shards with steps.

Build it **once, locally**, and upload it to Kaggle as a Dataset — `/kaggle/working` is
wiped when a session ends, and these runs span several sessions, so downloading inside the
training notebook re-pays the download and tokenization every time.

```bash
# 1. build the shards OUTSIDE any cloud-synced folder
python fineweb.py --shards 7 --local_dir C:/data/edu_fineweb10B

# 2. upload as a Kaggle Dataset (the CLI handles ~GB better than the web uploader)
pip install kaggle                       # token: kaggle.com/settings -> Create New API Token
                                         # save to ~/.kaggle/kaggle.json
kaggle datasets init -p C:/data/edu_fineweb10B
#    edit dataset-metadata.json: set "title" and "id" to <username>/fineweb-edu-700m-gpt2
kaggle datasets create -p C:/data/edu_fineweb10B
```

Then in `kaggle_train.ipynb`, **Add Input** -> your dataset. It mounts read-only under
`/kaggle/input`, persists across sessions, and costs no session time; the notebook's data
cell finds it automatically and skips the download.

For a smoke test with no download at all, pass `--data_dir=input.txt` to the trainer.

### Kaggle (2× T4)

Open [`kaggle_train.ipynb`](kaggle_train.ipynb), set Accelerator to **GPU T4 x2** and
Internet to **On**, and run it top to bottom. It runs the correctness checks, a 30-second
smoke test, the data download, then trains the ablations **one model per GPU** so two
architectures train at once — each pinned with `CUDA_VISIBLE_DEVICES`, with a small
scheduler handing the next queued run to whichever GPU frees up first. This is not DDP:
DDP makes both GPUs cooperate on one model, which is the wrong tool when the point is
comparing architectures. The notebook finishes by plotting the loss curves and printing
the comparison table.

Two Kaggle-specific details are handled in code, and both bite if ignored:

- **T4 is SM 7.5 and has no bf16.** `train_gpt2.py` hardcodes `torch.bfloat16`. The modern
  trainer picks the dtype from the hardware and enables a `GradScaler` when it lands on
  fp16 — fp16 has bf16's 10-bit mantissa but far less exponent range, so small gradients
  flush to zero without loss scaling. Test that path on a bf16-capable card first with
  `--dtype=float16`.
- **Sessions die at ~12h; quota is 30h/week.** Runs checkpoint to `log/<run>/ckpt.pt` and
  restart with `--resume`. Checkpoints are written at the *end* of a step, so resuming at
  `step+1` neither repeats nor drops an optimizer update — verified by confirming that a
  6-then-resume-to-9 run reproduces a continuous 9-step run's losses exactly.

The equivalent from a shell, if you would rather not use the notebook:

```bash
# one model per GPU, two at a time
CUDA_VISIBLE_DEVICES=0 python train_modern_gpt.py --pos=learned --norm=layernorm --resume &
CUDA_VISIBLE_DEVICES=1 python train_modern_gpt.py --pos=rope    --norm=layernorm --resume &
wait
CUDA_VISIBLE_DEVICES=0 python train_modern_gpt.py --pos=rope    --norm=rmsnorm   --resume

# or both GPUs on a single model (DDP) when you want one run to finish faster
torchrun --standalone --nproc_per_node=2 train_modern_gpt.py --pos=rope --norm=rmsnorm
```

Defaults are a ~30M model (6L / 6H / 384d, 512 ctx), sized so three ablations fit inside
the weekly GPU quota. For the full GPT-2 124M configuration:

```bash
python train_modern_gpt.py --n_layer=12 --n_head=12 --n_embd=768 --block_size=1024 \
    --batch_size=8 --seq_len=1024 --total_batch_size=524288 --max_steps=19073 \
    --warmup_steps=715
```

Run `python train_modern_gpt.py --help` for the rest.

---

## Future work

Deliberately out of scope for this release: SwiGLU, GQA/MQA, long-context support
(YaRN/NTK scaling), speculative decoding, quantization, MoE, and DPO / instruction tuning.
The KV cache section above is where GQA would slot in most naturally, since cache size is
already the binding constraint.

## Credit

The GPT-2 reproduction in `train_gpt2.py` is from Andrej Karpathy's
[build-nanogpt](https://github.com/karpathy/build-nanogpt)
([video](https://youtu.be/l8pRSuU81PU)), kept unmodified as the control.

## License

MIT

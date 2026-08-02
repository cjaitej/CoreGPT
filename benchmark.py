"""
Benchmarks and correctness checks for train_modern_gpt.py.

    # KV cache on/off: tokens/sec, ms/token, peak VRAM
    python benchmark.py --mode=inference

    # training throughput and memory for one architecture
    python benchmark.py --mode=training --pos=rope --norm=rmsnorm

    # correctness: baseline parity, RoPE relative-position property,
    # and that the KV cache changes speed and nothing else
    python benchmark.py --mode=check

Add --ckpt=log/rope_rmsnorm/ckpt.pt to benchmark trained weights. Speed and
memory do not depend on the weights, so random init is fine for timing.
"""

import os
import time
import math
import argparse

import torch
import torch.nn as nn

from train_modern_gpt import GPT, GPTConfig

# -----------------------------------------------------------------------------

def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(device_type, requested):
    if requested != "auto":
        return getattr(torch, requested)
    if device_type != "cuda":
        return torch.float32
    # T4 (SM 7.5) has no bf16; fall back to fp16 rather than silently crawling
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def build_model(args, device):
    if args.ckpt:
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config)
        model.load_state_dict(ckpt["model"])
        print(f"loaded {args.ckpt} (step {ckpt['step']}, val loss {ckpt['val_loss']:.4f})")
    else:
        config = GPTConfig(
            block_size=args.block_size, vocab_size=args.vocab_size,
            n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd,
            pos=args.pos, norm=args.norm,
        )
        model = GPT(config)
    model.to(device).eval()
    return model, config


def sync(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()


def peak_memory_mb(device_type):
    return torch.cuda.max_memory_allocated() / 1e6 if device_type == "cuda" else float("nan")


def reset_memory(device_type):
    if device_type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

# -----------------------------------------------------------------------------

@torch.no_grad()
def time_generation(model, device, device_type, amp_dtype, batch_size,
                    prompt_len, gen_len, use_cache, warmup=1, repeats=3):
    """Time a full generate() call and report per-token cost and peak memory."""
    vocab = model.config.vocab_size
    idx = torch.randint(0, vocab, (batch_size, prompt_len), device=device)
    gen = torch.Generator(device=device).manual_seed(0)

    def run():
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(device_type == "cuda")):
            return model.generate(idx, gen_len, top_k=50, use_cache=use_cache,
                                  generator=gen, kv_dtype=amp_dtype)

    for _ in range(warmup):
        run()
    sync(device_type)
    reset_memory(device_type)

    times = []
    for _ in range(repeats):
        sync(device_type)
        t0 = time.perf_counter()
        run()
        sync(device_type)
        times.append(time.perf_counter() - t0)

    best = min(times)
    tokens = batch_size * gen_len
    cache_mb = float("nan")
    if use_cache:
        cache = model.new_kv_cache(batch_size, prompt_len + gen_len, device, amp_dtype)
        cache_mb = cache.memory_bytes() / 1e6
        del cache
    return {
        "seconds": best,
        "tok_per_sec": tokens / best,
        "ms_per_token": best / gen_len * 1000,  # wall-clock latency per decode step
        "peak_mb": peak_memory_mb(device_type),
        "cache_mb": cache_mb,
    }


def bench_inference(args, model, device, device_type, amp_dtype):
    rows = []
    for prompt_len, gen_len in args.shapes:
        if prompt_len + gen_len > model.config.block_size:
            print(f"skipping {prompt_len}+{gen_len}: exceeds block_size {model.config.block_size}")
            continue
        for use_cache in (False, True):
            r = time_generation(model, device, device_type, amp_dtype, args.batch_size,
                                prompt_len, gen_len, use_cache,
                                warmup=args.warmup, repeats=args.repeats)
            r.update(prompt_len=prompt_len, gen_len=gen_len, use_cache=use_cache)
            rows.append(r)
            print(f"  prompt={prompt_len:<5} gen={gen_len:<5} cache={str(use_cache):<5} "
                  f"{r['tok_per_sec']:8.1f} tok/s  {r['ms_per_token']:7.2f} ms/tok")

    print()
    print(f"| Prompt | Gen | KV cache | Tok/s | ms/token | Peak VRAM (MB) | Cache (MB) | Speedup |")
    print(f"| ---: | ---: | :---: | ---: | ---: | ---: | ---: | ---: |")
    baseline = {}
    for r in rows:
        key = (r["prompt_len"], r["gen_len"])
        if not r["use_cache"]:
            baseline[key] = r["tok_per_sec"]
        speedup = r["tok_per_sec"] / baseline[key] if key in baseline else float("nan")
        cache_mb = "-" if math.isnan(r["cache_mb"]) else f"{r['cache_mb']:.1f}"
        print(f"| {r['prompt_len']} | {r['gen_len']} | {'yes' if r['use_cache'] else 'no'} "
              f"| {r['tok_per_sec']:.1f} | {r['ms_per_token']:.2f} | {r['peak_mb']:.1f} "
              f"| {cache_mb} | {speedup:.2f}x |")
    return rows

# -----------------------------------------------------------------------------

def bench_training(args, model, device, device_type, amp_dtype):
    """Step time, throughput and peak memory for one architecture."""
    B, T = args.batch_size, min(args.seq_len, model.config.block_size)
    model.train()
    optimizer = model.configure_optimizers(0.1, 6e-4, device_type, verbose=False)
    use_scaler = (amp_dtype == torch.float16)
    scaler = torch.amp.GradScaler(device_type, enabled=use_scaler)
    x = torch.randint(0, model.config.vocab_size, (B, T), device=device)
    y = torch.randint(0, model.config.vocab_size, (B, T), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device_type, dtype=amp_dtype, enabled=(device_type == "cuda")):
            _, loss = model(x, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

    for _ in range(args.warmup):
        step()
    sync(device_type)
    reset_memory(device_type)

    times = []
    for _ in range(args.repeats):
        sync(device_type)
        t0 = time.perf_counter()
        step()
        sync(device_type)
        times.append(time.perf_counter() - t0)

    dt = sum(times) / len(times)
    n_params = sum(p.numel() for p in model.parameters())
    print()
    print(f"| Arch | Params | B x T | ms/step | Tok/s | Peak VRAM (MB) |")
    print(f"| :--- | ---: | :---: | ---: | ---: | ---: |")
    print(f"| {args.pos} + {args.norm} | {n_params:,} | {B}x{T} | {dt*1000:.1f} "
          f"| {B*T/dt:.0f} | {peak_memory_mb(device_type):.1f} |")

# -----------------------------------------------------------------------------

@torch.no_grad()
def check_cache(args, model, device):
    """The cache is only a valid optimization if it is a no-op numerically.

    Forward the whole sequence at once, then replay it one token at a time
    through the cache, and compare logits at every position.
    """
    model.float().eval()  # fp32: we are checking math, not speed
    B, T = 2, min(64, model.config.block_size)
    idx = torch.randint(0, model.config.vocab_size, (B, T), device=device)

    ref_logits, _ = model(idx)

    cache = model.new_kv_cache(B, T, device, torch.float32)
    got = [model(idx[:, :1], kv_cache=cache)[0]]
    for t in range(1, T):
        got.append(model(idx[:, t:t + 1], kv_cache=cache)[0])
    got_logits = torch.cat(got, dim=1)

    diff = (ref_logits - got_logits).abs().max().item()
    scale = ref_logits.abs().max().item()
    print(f"arch: pos={model.config.pos} norm={model.config.norm}")
    print(f"max |no-cache - cached| logit difference: {diff:.3e} (logit scale {scale:.3f})")
    ok = diff < 1e-3 * max(scale, 1.0)
    print("PASS: KV cache is numerically equivalent" if ok else "FAIL: cache changes the output")

    return ok


@torch.no_grad()
def check_rope_relative(device):
    """RoPE's whole claim: <R(m)q, R(n)k> depends only on n - m.

    Three query/key pairs placed at absolute positions 3/7, 10/14 and 50/54 all
    have offset +4, so all three must produce the same attention score.
    """
    from train_modern_gpt import RotaryEmbedding, apply_rotary
    head_dim = 32
    rope = RotaryEmbedding(head_dim, 128).to(device)
    q = torch.randn(1, 1, 1, head_dim, device=device)
    k = torch.randn(1, 1, 1, head_dim, device=device)

    def score(m, n):
        cq, sq = rope(m, 1)
        ck, sk = rope(n, 1)
        qr, _ = apply_rotary(q, q, cq, sq)
        kr, _ = apply_rotary(k, k, ck, sk)
        return (qr * kr).sum().item()

    same = [score(3, 7), score(10, 14), score(50, 54)]   # offset +4
    other = score(3, 9)                                   # offset +6
    spread = max(same) - min(same)
    ok = spread < 1e-4 and abs(same[0] - other) > 1e-3
    print(f"offset +4 at absolute 3/10/50: {same[0]:.6f}, {same[1]:.6f}, {same[2]:.6f} "
          f"(spread {spread:.2e})")
    print(f"offset +6 for contrast: {other:.6f}")
    print("PASS: RoPE scores are translation invariant" if ok
          else "FAIL: RoPE scores depend on absolute position")
    return ok


@torch.no_grad()
def check_baseline_parity():
    """--pos=learned --norm=layernorm must BE the original nanoGPT model.

    Without this the baseline row of the comparison table is just a different
    random model, and any loss gap could be the seed rather than the
    architecture. train_gpt2.py runs its training loop at import time, so we
    exec only the part above the data-loading section to get at its classes.
    """
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_gpt2.py"),
               encoding="utf-8").read()
    head = src.split("import tiktoken")[0]
    ns = {"__name__": "_orig_gpt2", "master_process": False}
    exec(compile(head, "train_gpt2.py", "exec"), ns)

    small = dict(block_size=64, vocab_size=512, n_layer=2, n_head=4, n_embd=64)
    torch.manual_seed(1337)
    ref = ns["GPT"](ns["GPTConfig"](**small)).eval()
    torch.manual_seed(1337)
    new = GPT(GPTConfig(**small, pos="learned", norm="layernorm")).eval()

    sd_r, sd_n = ref.state_dict(), new.state_dict()
    if set(sd_r) != set(sd_n):
        print(f"FAIL: parameter names differ: {set(sd_r) ^ set(sd_n)}")
        return False
    wdiff = max((sd_r[k] - sd_n[k]).abs().max().item() for k in sd_r)
    idx = torch.randint(0, small["vocab_size"], (2, 32))
    ldiff = (ref(idx)[0] - new(idx)[0]).abs().max().item()
    ok = wdiff == 0.0 and ldiff == 0.0
    print(f"{len(sd_r)} tensors, max weight diff {wdiff:.1e}, max logit diff {ldiff:.1e}")
    print("PASS: baseline is bit-identical to train_gpt2.py" if ok
          else "FAIL: baseline has drifted from train_gpt2.py")
    return ok


def run_checks(args, model, device):
    results = []
    print("--- baseline parity (train_gpt2.py vs --pos=learned --norm=layernorm) ---")
    results.append(check_baseline_parity())
    print("\n--- RoPE relative position property ---")
    results.append(check_rope_relative(device))
    print("\n--- KV cache equivalence ---")
    results.append(check_cache(args, model, device))
    print()
    print("ALL CHECKS PASSED" if all(results) else "SOME CHECKS FAILED")
    return all(results)

# -----------------------------------------------------------------------------

def parse_shapes(s):
    """'128:128,512:256' -> [(128, 128), (512, 256)]"""
    out = []
    for part in s.split(","):
        a, b = part.split(":")
        out.append((int(a), int(b)))
    return out


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["inference", "training", "check"], default="inference")
    p.add_argument("--ckpt", default=None, help="checkpoint to load instead of random init")
    # architecture (ignored when --ckpt is given)
    p.add_argument("--pos", choices=["learned", "rope"], default="rope")
    p.add_argument("--norm", choices=["layernorm", "rmsnorm"], default="rmsnorm")
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=384)
    p.add_argument("--block_size", type=int, default=512)
    p.add_argument("--vocab_size", type=int, default=50304)
    # measurement
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=512, help="training mode only")
    p.add_argument("--shapes", type=parse_shapes, default="64:128,128:256,256:256",
                   help="prompt_len:gen_len pairs, comma separated")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    return p.parse_args()


def main():
    args = get_args()
    if isinstance(args.shapes, str):
        args.shapes = parse_shapes(args.shapes)
    device = pick_device()
    device_type = "cuda" if device.startswith("cuda") else device
    amp_dtype = pick_dtype(device_type, args.dtype)

    model, config = build_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device} | dtype: {amp_dtype} | params: {n_params:,}")
    print(f"arch: pos={config.pos} norm={config.norm} "
          f"L={config.n_layer} H={config.n_head} D={config.n_embd}")
    if device_type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
    print()

    if args.mode == "inference":
        bench_inference(args, model, device, device_type, amp_dtype)
    elif args.mode == "training":
        bench_training(args, model, device, device_type, amp_dtype)
    else:
        raise SystemExit(0 if run_checks(args, model, device) else 1)


if __name__ == "__main__":
    main()

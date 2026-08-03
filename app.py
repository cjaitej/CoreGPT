import gc
import glob
import html
import math
import os
import sys
import time

import streamlit as st
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train_modern_gpt import GPT, GPTConfig, pick_amp_dtype

SLACK = 60
STREAM_EVERY = 8
ACCENTS = ["#5b9bf8", "#f2994a", "#4fbf8b", "#c77dff"]

PROMPT = (
    "Photosynthesis is the process by which green plants, algae, and certain bacteria "
    "convert light energy into chemical energy. Inside the chloroplast, chlorophyll "
    "absorbs sunlight and uses that energy to split water molecules, releasing oxygen "
    "as a by-product. The hydrogen that remains is combined with carbon dioxide drawn "
    "from the air to build sugars, which the organism then uses to"
)

BENCH_PROMPT = (
    "The Industrial Revolution began in Britain during the late eighteenth century, "
    "when a series of inventions transformed the way goods were produced. Steam power "
    "replaced water and muscle, factories replaced workshops, and cities grew rapidly "
    "around the new centres of manufacturing."
)

CSS = """
<style>
div.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1500px; }
#MainMenu, footer { visibility: hidden; }

.masthead { display:flex; align-items:baseline; justify-content:space-between;
            gap:16px; padding-bottom:10px; margin-bottom:18px;
            border-bottom:1px solid rgba(128,128,128,.22); flex-wrap:wrap; }
.masthead h1 { font-size:1.5rem; font-weight:650; margin:0; letter-spacing:-.02em; }
.hw { font-size:.72rem; opacity:.6; font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
      border:1px solid rgba(128,128,128,.28); border-radius:999px; padding:4px 11px; }

.head { display:flex; align-items:center; gap:9px; margin-bottom:5px; }
.dot { width:9px; height:9px; border-radius:50%; flex:none; }
.name { font-weight:600; font-size:.95rem; }
.chips { margin-bottom:10px; }
.chip { display:inline-block; padding:2px 9px; margin:0 5px 5px 0; border-radius:999px;
        font-size:.7rem; letter-spacing:.02em; background:rgba(128,128,128,.14);
        border:1px solid rgba(128,128,128,.18); }

.card { border:1px solid rgba(128,128,128,.22); border-left-width:3px; border-radius:9px;
        padding:15px 17px; background:rgba(128,128,128,.05);
        font-family:Georgia,'Iowan Old Style',serif; font-size:.94rem; line-height:1.7; }
.card .given { opacity:.45; }
.card .made { font-weight:500; }
.tag { font-size:.68rem; letter-spacing:.07em; text-transform:uppercase;
       opacity:.5; margin-bottom:5px; }

.stats { display:flex; gap:26px; margin:12px 0 4px; flex-wrap:wrap; }
.stats b { display:block; font-size:1.1rem; font-weight:620; line-height:1.25;
           font-variant-numeric:tabular-nums; }
.stats span { font-size:.68rem; letter-spacing:.05em; text-transform:uppercase; opacity:.5; }

.note { font-size:.78rem; opacity:.6; line-height:1.55; margin:10px 0 4px; }
.match { font-size:.78rem; color:#4fbf8b; margin:10px 0 4px; }
.rule { height:1px; background:rgba(128,128,128,.2); margin:26px 0 16px; }
section[data-testid="stSidebar"] hr { margin:14px 0; }
</style>
"""

st.set_page_config(page_title="CoreGPT", page_icon="⚡", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def encoder():
    import tiktoken
    return tiktoken.get_encoding("gpt2")


@st.cache_resource(show_spinner=False)
def load_checkpoint(path, name, device, weight_dtype):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**ckpt["config"])
    model = GPT(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device=device, dtype=weight_dtype)
    info = {
        "name": name,
        "pos": cfg.pos,
        "norm": cfg.norm,
        "params": sum(p.numel() for p in model.parameters()),
        "step": ckpt.get("step"),
        "val_loss": ckpt.get("val_loss"),
        "ctx": cfg.block_size,
    }
    return model, info


@st.cache_resource(show_spinner=False)
def untrained(pos, norm, device, weight_dtype):
    cfg = GPTConfig(block_size=512, vocab_size=50304, n_layer=6, n_head=6,
                    n_embd=384, pos=pos, norm=norm)
    model = GPT(cfg).eval().to(device=device, dtype=weight_dtype)
    info = {"name": f"untrained {pos}/{norm}", "pos": pos, "norm": norm,
            "params": sum(p.numel() for p in model.parameters()),
            "step": None, "val_loss": None, "ctx": cfg.block_size}
    return model, info


def cpu_budget():
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return int(quota) / int(period)
    except OSError:
        pass
    try:
        quota = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        period = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if quota > 0:
            return quota / period
    except OSError:
        pass
    return float(os.cpu_count() or 1)


def autocast(device, dtype):
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=device == "cuda")


def pick_next(logits, temperature, top_k, gen):
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = F.softmax(logits / temperature, dim=-1)
    if not top_k:
        return torch.multinomial(probs, 1, generator=gen)
    values, indices = torch.topk(probs, min(top_k, probs.size(-1)), dim=-1)
    return torch.gather(indices, -1, torch.multinomial(values, 1, generator=gen))


@torch.no_grad()
def stream(model, ids, limit, temperature, top_k, cached, device, dtype, seed):
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None]
    gen = torch.Generator(device=device).manual_seed(seed)
    kv = None
    if cached:
        kv = model.new_kv_cache(1, idx.size(1) + limit, device, KV_DTYPE)
    for step in range(limit):
        if kv is None:
            window = idx[:, -model.config.block_size:]
        else:
            window = idx if step == 0 else idx[:, -1:]
        with autocast(device, dtype):
            logits, _ = model(window, kv_cache=kv, last_only=True)
        nxt = pick_next(logits[:, -1, :].float(), temperature, top_k, gen)
        idx = torch.cat([idx, nxt], dim=1)
        yield nxt.item()


def complete(text):
    tail = text.rstrip().rstrip("\"')]}»”’")
    return bool(tail) and tail[-1] in ".!?"


def clip(text):
    end = max(text.rfind(mark) for mark in ".!?")
    return text[:end + 1] if end > 0 else text


@torch.no_grad()
def perplexity(model, ids, device, dtype):
    ids = ids[-model.config.block_size:]
    if len(ids) < 2:
        return float("nan")
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None]
    with autocast(device, dtype):
        _, loss = model(idx[:, :-1], idx[:, 1:])
    return math.exp(loss.item())


@torch.no_grad()
def time_run(model, ids, batch, count, temperature, top_k, cached, device, dtype, seed):
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None].repeat(batch, 1)
    kwargs = dict(temperature=max(temperature, 1e-6), top_k=top_k, use_cache=cached,
                  generator=torch.Generator(device=device).manual_seed(seed),
                  kv_dtype=KV_DTYPE)
    with autocast(device, dtype):
        model.generate(idx, min(16, count), **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with autocast(device, dtype):
        out = model.generate(idx, count, **kwargs)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    peak = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else float("nan")
    return batch * count / elapsed, elapsed / count * 1000, peak, out[0, len(ids):]


def escape(text):
    return html.escape(text).replace("\n", "<br>")


def card(given, made, accent, tag=None):
    label = f'<div class="tag">{tag}</div>' if tag else ""
    return (f'{label}<div class="card" style="border-left-color:{accent}">'
            f'<span class="given">{escape(given)}</span>'
            f'<span class="made">{escape(made)}</span></div>')


def heading(name, accent, chips):
    tags = "".join(f'<span class="chip">{html.escape(c)}</span>' for c in chips)
    return (f'<div class="head"><span class="dot" style="background:{accent}"></span>'
            f'<span class="name">{html.escape(name)}</span></div>'
            f'<div class="chips">{tags}</div>')


def stats(pairs):
    cells = "".join(f"<div><b>{v}</b><span>{k}</span></div>" for k, v in pairs)
    return f'<div class="stats">{cells}</div>'


device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    WEIGHT_DTYPE, dtype = torch.float32, pick_amp_dtype(device)
else:
    WEIGHT_DTYPE, dtype = torch.bfloat16, torch.bfloat16
USE_CACHE = True
KV_DTYPE = dtype if device == "cuda" else WEIGHT_DTYPE

if device == "cuda":
    hardware = torch.cuda.get_device_name(0)
    BENCH_BATCH, BENCH_TOKENS = 16, 160
else:
    vcpu = cpu_budget()
    torch.set_num_threads(max(1, int(vcpu)))
    hardware = f"CPU · {vcpu:g} vCPU · {torch.get_num_threads()} threads"
    BENCH_BATCH, BENCH_TOKENS = 4, 64

st.markdown(
    f'<div class="masthead"><h1>CoreGPT</h1>'
    f'<div class="hw">{html.escape(hardware)} &middot; {str(dtype).split(".")[-1]}</div></div>',
    unsafe_allow_html=True)

found = {os.path.splitext(os.path.basename(p))[0]: p
         for p in sorted(glob.glob(os.path.join(ROOT, "models", "*.pt")))}
if not found:
    found = {os.path.basename(os.path.dirname(p)): p
             for p in sorted(glob.glob(os.path.join(ROOT, "log", "*", "ckpt.pt")))}

if found:
    names = st.sidebar.multiselect("Checkpoints", list(found), default=list(found)[:2])
    loaded = [load_checkpoint(found[name], name, device, WEIGHT_DTYPE) for name in names]
else:
    st.sidebar.caption("No weights in models/*.pt or log/*/ckpt.pt — using untrained models.")
    names = st.sidebar.multiselect("Untrained", ["rope/rmsnorm", "learned/layernorm"],
                                   default=["rope/rmsnorm"])
    loaded = [untrained(*name.split("/"), device, WEIGHT_DTYPE) for name in names]

st.sidebar.divider()
limit = st.sidebar.slider("Max new tokens", 8, 400, 120, 8)
temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.8, 0.05)
top_k = st.sidebar.slider("Top-k", 0, 200, 50, 5)
seed = st.sidebar.number_input("Seed", value=42, step=1)
finish = st.sidebar.checkbox("Finish the sentence", value=True)

if not loaded:
    st.caption("Select a model in the sidebar.")
    st.stop()

enc = encoder()
gen_tab, cache_tab, model_tab = st.tabs(["Generate", "KV cache", "Models"])

with gen_tab:
    prompt = st.text_area("Prompt", PROMPT, height=150, label_visibility="collapsed")
    run, hint = st.columns([1, 5])
    go = run.button("Generate", type="primary", width="stretch")
    hint.markdown(
        '<div class="note">Every model decodes with the KV cache on, so the speed gap '
        'here is the cost of RoPE and RMSNorm. The cache itself is measured in the next '
        'tab.</div>', unsafe_allow_html=True)

    if go:
        ids = enc.encode(prompt) if prompt.strip() else [enc.eot_token]

        for slot_index, (column, (model, info)) in enumerate(
                zip(st.columns(len(loaded)), loaded)):
            accent = ACCENTS[slot_index % len(ACCENTS)]
            room = info["ctx"] - len(ids)
            count = min(limit, room)
            with column:
                chips = [info["pos"], info["norm"], f"{info['params']:,} params"]
                if info["val_loss"]:
                    chips.append(f"val ppl {math.exp(info['val_loss']):.1f}")
                st.markdown(heading(info["name"], accent, chips), unsafe_allow_html=True)

                if count <= 0:
                    st.error(f"Prompt is {len(ids)} tokens, context is {info['ctx']}.")
                    continue

                extra = min(SLACK, room - count) if finish else 0
                target = st.empty()
                out, text = [], ""
                start = time.perf_counter()
                for i, token in enumerate(stream(model, ids, count + extra, temperature,
                                                 top_k or None, USE_CACHE, device, dtype,
                                                 int(seed))):
                    out.append(token)
                    at_limit = i + 1 >= count
                    if i % STREAM_EVERY == 0 or at_limit:
                        text = enc.decode(out)
                        target.markdown(card(prompt, text, accent), unsafe_allow_html=True)
                        if finish and at_limit and complete(text):
                            break
                elapsed = time.perf_counter() - start

                text = enc.decode(out)
                if finish and not complete(text):
                    text = clip(text)
                target.markdown(card(prompt, text, accent), unsafe_allow_html=True)

                ppl = perplexity(model, ids, device, dtype)
                st.markdown(stats([
                    (f"prompt ppl · {len(ids)} tok", f"{ppl:.1f}" if ppl == ppl else "—"),
                    ("avg tok/s", f"{len(out) / elapsed:.0f}"),
                    ("avg ms/token", f"{elapsed / len(out) * 1000:.1f}"),
                ]), unsafe_allow_html=True)

with cache_tab:
    bench_prompt = st.text_area("Prompt", BENCH_PROMPT, height=110,
                                key="bench_prompt", label_visibility="collapsed")
    left, right, action = st.columns([2, 2, 1])
    count = left.slider("Tokens", 32, 400, BENCH_TOKENS, 8, key="bench_tokens")
    batch = right.select_slider("Batch", [1, 2, 4, 8, 16, 32], value=BENCH_BATCH,
                                key="bench_batch")
    action.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    start_bench = action.button("Run", type="primary", width="stretch")

    prompt_len = len(enc.encode(bench_prompt)) if bench_prompt.strip() else 1
    work = batch * count * (prompt_len + count / 2) * len(loaded)

    if device != "cuda" and work > 60_000:
        st.markdown('<div class="note">⚠ This will take a while on CPU. Drop the batch '
                    'or token count for a quicker result — the trend holds at any '
                    'size.</div>', unsafe_allow_html=True)

    if start_bench:
        ids = enc.encode(bench_prompt) if bench_prompt.strip() else [enc.eot_token]
        speed, samples = [], []
        bar = st.progress(0.0)

        for i, (model, info) in enumerate(loaded):
            n = min(count, info["ctx"] - len(ids))
            if n <= 0:
                continue
            runs = {}
            for j, on in enumerate((False, True)):
                runs[on] = time_run(model, ids, batch, n, temperature, top_k or None,
                                    on, device, dtype, int(seed))
                bar.progress((i * 2 + j + 1) / (len(loaded) * 2))
            for on in (False, True):
                tok_s, ms, peak, _ = runs[on]
                speed.append({
                    "Model": info["name"],
                    "Batch": batch,
                    "Cache": "yes" if on else "no",
                    "Tok/s": round(tok_s, 1),
                    "ms/step": round(ms, 2),
                    "Peak VRAM (MB)": round(peak, 1) if device == "cuda" else None,
                    "Speedup": round(runs[True][0] / runs[False][0], 2) if on else 1.0,
                })
            samples.append((info["name"],
                            enc.decode(runs[False][3].tolist()),
                            enc.decode(runs[True][3].tolist())))

        bar.empty()
        st.dataframe(speed, width="stretch", hide_index=True)
        st.markdown('<div class="note">With the cache, ms/step stays flat as batch grows. '
                    'Without it, the whole prefix is recomputed for every token of every '
                    'sequence.</div>', unsafe_allow_html=True)

        for slot_index, (name, without, with_cache) in enumerate(samples):
            accent = ACCENTS[slot_index % len(ACCENTS)]
            if slot_index:
                st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="head"><span class="dot" style="background:{accent}">'
                        f'</span><span class="name">{html.escape(name)}</span></div>',
                        unsafe_allow_html=True)
            off_col, on_col = st.columns(2)
            off_col.markdown(card(bench_prompt, without, accent, "cache off"),
                             unsafe_allow_html=True)
            on_col.markdown(card(bench_prompt, with_cache, accent, "cache on"),
                            unsafe_allow_html=True)
            if without == with_cache:
                st.markdown('<div class="match">Identical output.</div>',
                            unsafe_allow_html=True)
            else:
                st.markdown('<div class="note">The two differ. Under fp16 a rounding '
                            'difference of ~1e-3 in one logit flips a single sampled '
                            'token, and everything after it diverges. Set temperature to '
                            '0 for an exact match.</div>', unsafe_allow_html=True)

with model_tab:
    st.dataframe([{
        "Run": info["name"],
        "Position": info["pos"],
        "Norm": info["norm"],
        "Params": f"{info['params']:,}",
        "Step": info["step"],
        "Val loss": round(info["val_loss"], 4) if info["val_loss"] else None,
        "PPL": round(math.exp(info["val_loss"]), 2) if info["val_loss"] else None,
        "Context": info["ctx"],
    } for _, info in loaded], width="stretch", hide_index=True)

    if st.button("Reload models" if device == "cuda" else "Free memory"):
        st.cache_resource.clear()
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        st.rerun()

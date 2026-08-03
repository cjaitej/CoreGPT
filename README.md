# Modern nanoGPT

Modernizing GPT-2 with Llama-inspired architectural changes, measured one at a time.

Three changes to a from-scratch GPT-2 reproduction, each isolated in its own training run
so the effect of every one is separately attributable:

| Change       | What it replaces                                                    | Retrain? |
| :----------- | :------------------------------------------------------------------ | :------: |
| **RoPE**     | Learned position embeddings → rotate Q/K by position in every layer |   yes    |
| **RMSNorm**  | LayerNorm → drop mean-centering and the bias                        |   yes    |
| **KV cache** | Recomputing the prefix each step → store K/V, decode one token      |    no    |

```
train_gpt2.py          original GPT-2, unmodified (the control)
train_modern_gpt.py    flag-gated model + training loop
benchmark.py           inference / training / correctness harness
app.py                 Streamlit demo
kaggle_train.ipynb     runs all three ablations
```

Architecture is selected by flags, so one file produces every row of the table:

```bash
python train_modern_gpt.py --pos=learned --norm=layernorm   # baseline
python train_modern_gpt.py --pos=rope    --norm=layernorm   # + RoPE
python train_modern_gpt.py --pos=rope    --norm=rmsnorm     # + RMSNorm
```

---

## Results

30M params (6 layers, 6 heads, 384 dim, 512 context), 4000 steps × 131,072 tokens =
**524M tokens** of FineWeb-Edu per run. Identical seed, data order and LR schedule across
all three; only `--pos` and `--norm` differ. Trained on a Kaggle T4.

| Model    | Position | Norm      |     Params |   Val loss |       PPL |           Δ | Train tok/s |
| :------- | :------- | :-------- | ---------: | ---------: | --------: | ----------: | ----------: |
| baseline | learned  | LayerNorm | 30,160,896 |     4.0489 |     57.33 |           — |      54,830 |
| + RoPE   | rope     | LayerNorm | 29,964,288 | **3.9251** | **50.66** | **−0.1238** |      40,425 |
| modern   | rope     | RMSNorm   | 29,959,296 |     3.9277 |     50.79 |     −0.1212 |      45,259 |

### KV cache, by batch size

Modern model, 200 generated tokens, RTX 3050:

| Batch |     cache off | cache on |   Speedup |
| ----: | ------------: | -------: | --------: |
|     1 | 11.87 ms/step |    12.01 |     0.99x |
|     4 |         13.40 |    15.95 | **0.84x** |
|     8 |         41.46 |    16.58 |     2.50x |
|    16 |         80.23 |    15.58 | **5.15x** |

### KV cache, by context length

Batch 8, bf16, RTX 3050:

| Prompt | Gen | Cache | Tok/s | ms/token | Peak VRAM (MB) | Cache (MB) |   Speedup |
| -----: | --: | :---: | ----: | -------: | -------------: | ---------: | --------: |
|     64 | 128 |  no   | 529.9 |    15.10 |          355.1 |          – |     1.00x |
|     64 | 128 |  yes  | 815.8 |     9.81 |          262.1 |       14.2 |     1.54x |
|    128 | 256 |  no   | 278.8 |    28.69 |          516.6 |          – |     1.00x |
|    128 | 256 |  yes  | 775.1 |    10.32 |          328.7 |       28.3 |     2.78x |
|    256 | 256 |  no   | 266.8 |    29.98 |          622.5 |          – |     1.00x |
|    256 | 256 |  yes  | 756.2 |    10.58 |          447.1 |       37.7 | **2.83x** |

### Correctness

```bash
python benchmark.py --mode=check
```

| Property                                     | Result                                                      |
| :------------------------------------------- | :---------------------------------------------------------- |
| Baseline is bit-identical to `train_gpt2.py` | max weight diff **0.0e+00**, max logit diff **0.0e+00**     |
| RoPE scores depend only on relative offset   | spread **1.9e-06** across absolute positions 3/10/50        |
| KV cache changes speed and nothing else      | **5.7e-06 – 1.4e-05** max logit diff on trained checkpoints |

The parity check is what makes the rest meaningful: submodules are constructed in the same
order as the original, so under one seed `--pos=learned --norm=layernorm` produces
bit-identical weights. Every gap in the table is architecture, not RNG.

---

## Findings

**1. RoPE delivers all of the quality gain; RMSNorm delivers none.**
RoPE alone: −0.124 nats, 11.6% lower perplexity, while _removing_ 196,608 parameters.
Adding RMSNorm moves val loss by +0.0026 — noise.

**2. That is the correct outcome for RMSNorm, not a failure.**
It was never a quality proposal. It is a simplification that costs nothing, and the
measurement shows exactly that: quality flat, **throughput +12%** (40.4K → 45.3K tok/s),
4,992 fewer parameters, one less reduction pass per norm. The reason to adopt it is that
it is free.

**3. RoPE is not free.** −26% training throughput against the baseline, because the
rotation is applied to Q and K in _every layer_, where a learned embedding is one lookup
for the whole forward pass. Net for the modern model: 11.6% better perplexity for 17% less
throughput. The same cost appears at inference — 41 vs 54 tok/s at batch 1, an independent
measurement agreeing with the training figure.

**4. The KV cache is a throughput optimization, not a latency one — below batch 8 it is
slower.** This is the most counter-intuitive result here, and "add a KV cache, get faster
generation" is repeated everywhere without it. Read the cache-on column above: **12.0 →
15.6 ms/step while batch grows 16x.** Single-token decoding is bound by kernel _launches_,
not arithmetic, so the GPU sits idle and sixteen sequences cost about what one does. The
cache-off column goes 11.9 → 80.2 ms, because it recomputes the whole prefix for every
token of every sequence, and that is real work that scales. The cache's benefit is
proportional to work it removes; its cost is a fixed per-token overhead paid regardless. At
batch 1 there is not enough in flight for the O(n²) term to cost anything. The honest
framing: **it converts a compute-bound O(n²) decode into a launch-bound O(n) decode, buying
throughput at constant latency** — what serving concurrent users needs, and why it does
nothing for one interactive session.

**5. `torch.cuda.is_bf16_supported()` lies on Turing, and it cost 3.4x.**
It defaults to `including_emulation=True`, so on a T4 (SM 7.5) the compute-capability test
fails and it falls through to an emulation check that succeeds and returns `True`.
Selecting bf16 on that basis puts every matmul on a path with no tensor cores: **~9K tok/s,
against ~31K after switching to fp16**, on hardware whose fp16 tensor cores were sitting
idle the whole time. `pick_amp_dtype()` checks compute capability directly — bf16 hardware
starts at SM 8.0.

**6. Small models are vocab-dominated.** With `n_embd=384` against a 50,304-token
vocabulary, `lm_head` is ~65% of total FLOPs. Throughput intuitions calibrated on parameter
count will be wrong, and it is why `--compile` (which fuses the cross-entropy) is the
largest remaining optimization.

**7. Perplexity on a short prompt ranks models backwards.** On a 7-token prompt the
baseline scores 12.7 and the modern model 16.5 — the reverse of their validation ranking.
On a 392-token passage it flips to 104.5 vs 99.5, matching. Six predictions is noise; the
app's default prompt is 73 tokens for this reason, where it reproduces the ablation
correctly (38.0 / 30.8 / 30.8).

---

## Running the app

```bash
pip install streamlit
streamlit run app.py
```

Discovers every checkpoint under `log/*/ckpt.pt` and runs any subset **side by side, same
prompt, same seed**.

- **Generate** — streaming output per model, with prompt perplexity, average tok/s and
  average ms/token. "Finish the sentence" lets generation run past the token limit (up to
  60 extra) to reach a sentence boundary instead of stopping mid-clause.
- **KV cache benchmark** — identical decode with and without the cache, within one model,
  with a batch-size slider to find the crossover. Correctness is judged on fp32 logits, not
  on sampled text: under fp16 with temperature > 0 a ~1e-3 logit difference flips one
  `multinomial` draw and the continuations diverge forever, which looks like corruption but
  is only sampling.
- **Models** — config, parameter count, step and val loss per checkpoint.

Parameter counts dedupe the tied `wte`/`lm_head` weight; summing `state_dict` naively
double-counts the 50304 × 384 embedding and over-reports by 19.3M.

## Reproducing

```bash
python fineweb.py --shards 9              # ~900M tokens of FineWeb-Edu
python benchmark.py --mode=check          # correctness
python benchmark.py --mode=inference      # KV cache tables
python benchmark.py --mode=training       # throughput per architecture
```

Training runs via [`kaggle_train.ipynb`](kaggle_train.ipynb), or directly with the three
flag combinations above. Checkpoints land in `log/<run>/` and resume with `--resume`.

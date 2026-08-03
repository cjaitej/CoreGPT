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

### KV cache

Measured with `lm_head` projecting **only the last position on both paths** — see finding 5.
A naive uncached baseline that projects the whole prefix every step makes the cache look
2-5x better than it is, because most of that gap is a wasted matmul rather than the cache.

GPU (RTX 3050), modern model, 160 new tokens:

| Batch | cache off | cache on | Speedup |
| ----: | --------: | -------: | ------: |
|     1 | 16.48 ms/step | 13.92 |   1.18x |
|     4 |         12.85 | 14.68 |   0.88x |
|     8 |         12.39 | 13.65 |   0.91x |
|    16 |         13.12 | 13.12 |   1.00x |
|    32 |         20.43 | 18.87 |   1.08x |

CPU (2 threads, bf16), batch 1:

| Prompt | Gen | cache off | cache on |    Speedup |
| -----: | --: | --------: | -------: | ---------: |
|     20 |  60 |  6.5 tok/s |     71.1 |  **11.0x** |
|     73 | 120 |       2.6 |     62.5 |  **24.3x** |
|    190 | 120 |       1.4 |     52.2 |  **38.5x** |

GPU, batch 8, by context length:

| Prompt | Gen | Cache | Tok/s | ms/token | Peak VRAM (MB) | Cache (MB) | Speedup |
| -----: | --: | :---: | ----: | -------: | -------------: | ---------: | ------: |
|     64 | 128 |  no   | 634.3 |    12.61 |          220.8 |          – |   1.00x |
|     64 | 128 |  yes  | 613.0 |    13.05 |          214.6 |       14.2 |   0.97x |
|    128 | 256 |  no   | 562.3 |    14.23 |          245.1 |          – |   1.00x |
|    128 | 256 |  yes  | 660.0 |    12.12 |          229.3 |       28.3 |   1.17x |
|    256 | 256 |  no   | 415.6 |    19.25 |          262.8 |          – |   1.00x |
|    256 | 256 |  yes  | 584.6 |    13.68 |          240.3 |       37.7 |   1.41x |

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

**4. Whether the KV cache helps at all depends entirely on whether you are compute-bound.**
On the GPU it does essentially nothing here — 0.88x to 1.18x across batch 1 to 32. On CPU
the same code is **11x to 38x faster**, and the gap widens with prompt length. The reason
is not the cache; it is what surrounds it. Decoding one token of a 30M model leaves an
RTX 3050 almost entirely idle, so the uncached path's extra work costs nothing in
wall-clock and the cache's per-step overhead is all you measure. A CPU has no such slack:
recomputing the prefix is real work, and removing it is the whole game. "Add a KV cache,
get faster generation" is repeated everywhere without this condition attached. It is a fix
for a compute bottleneck, and if you do not have one it buys you nothing.

**5. `torch.cuda.is_bf16_supported()` lies on Turing, and it cost 3.4x.**
It defaults to `including_emulation=True`, so on a T4 (SM 7.5) the compute-capability test
fails and it falls through to an emulation check that succeeds and returns `True`.
Selecting bf16 on that basis puts every matmul on a path with no tensor cores: **~9K tok/s,
against ~31K after switching to fp16**, on hardware whose fp16 tensor cores were sitting
idle the whole time. `pick_amp_dtype()` checks compute capability directly — bf16 hardware
starts at SM 8.0.

**6. Small models are vocab-dominated, and it corrupted an earlier measurement.**
With `n_embd=384` against a 50,304-token vocabulary, `lm_head` is ~65% of total FLOPs. The
generation loop was projecting *every* prefill position through it and discarding all but
the last row: **63.9 ms versus 6.2 ms** for the projection alone at a 91-token prompt, and
**883 ms versus 276 ms** for a whole prefill forward. Fixing it cut prefill latency 3.2x —
and, less comfortably, revealed that an earlier "5.15x KV cache speedup" was mostly this
waste rather than the cache, since the uncached path pays it on every single token. Profile
the baseline before you credit the optimization.

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

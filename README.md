# Ternary Whisper

**192 weight matrices of whisper-small — 198 180 864 weights, 81.98 % of the
model — hold nothing but −1, 0 and +1.** Normalised WER goes *down* relative to
stock whisper-small. Punctuation and multilinguality go *away*.

| LibriSpeech (greedy, batch 16) | this model | stock whisper-small |
|---|---|---|
| normalised WER, test-clean | **2.7145 %** | 3.3391 % |
| normalised WER, test-other | **7.3817 %** | 7.5002 % |
| long-form, 9 recordings | **3.632 %** | 3.891 % |
| utterances with a comma | 0.0–0.7 % | 63.3 % |
| utterances with a capital | 2 % | 98 % |

**Two caveats belong right here, not further down.**

*The comparison above is against stock whisper-small.* A matched fp16 control —
same code path, same data, same schedule, quantisation switched off — is still
**≈0.37 pp better** than the ternary branch. Both curves were still descending
when the runs ended, so the final quantisation gap is **not established**. This
model beats the teacher it was distilled from; that is not the same claim as
"ternarisation improved the model".

*The WER above is normalised* — case and punctuation are stripped before
scoring. So a model can score better and still produce worse text for a human,
and that is exactly what happened here.

### What "1.58 bits" does and does not mean

| | |
|---|---|
| information content of a ternary alphabet, log₂3 | 1.585 bit/weight |
| training layout: 2-bit code + fp16 scale per 128 | **2.125 bpw** |
| shipped file, ggml `Q2_0`, fp16 scale per 64 | **2.25 bpw** |
| whole 140 MiB file averaged over all parameters | **≈4.87 bit/param** |

The last row is higher because 18 % of the parameters — the tied embedding
head — plus convolutions, layer norms and biases stay in fp16. Of the 140 MiB,
**53 MiB is the ternary matrices and 87 MiB is everything else.**

The runtime is a hybrid too, deliberately:

```
on disk     encoder Q2_0        decoder Q2_0        head fp16
in RAM      encoder Q8_0 *      decoder Q2_0        head fp16
            * exact expansion — ternary codes and the same fp16 scale are
              representable in Q8_0 — chosen because the encoder is compute-bound
```

So the encoder is mathematically ternary and physically 8-bit while running.
Saying "the model runs at two bits" would be wrong.

This repository is the full recipe: the quantisation-aware training library, the
data pipeline, the exporter, the whisper.cpp patch, and every measurement file
the numbers above are read from.

- **Model weights:** [huggingface.co/armanibadboy/whisper-small-ternary](https://huggingface.co/armanibadboy/whisper-small-ternary)
- **Upstream patch:** [whisper.cpp Q2_0 support](deploy/whisper_cpp_q2_0.patch)

---

## Run it on a CPU

```bash
git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
git apply /path/to/whisper-ternary/deploy/whisper_cpp_q2_0.patch
cmake -B build && cmake --build build -j

hf download armanibadboy/whisper-small-ternary ggml-small-wal-ternary-q2_0.bin --local-dir models/
./build/bin/whisper-cli -m models/ggml-small-wal-ternary-q2_0.bin -f samples/jfk.wav -ng
```

As an HTTP service:

```bash
./build/bin/whisper-server -m models/ggml-small-wal-ternary-q2_0.bin \
    --host 127.0.0.1 --port 8178 -t 16 -ng -ml 100000

curl 127.0.0.1:8178/inference -F file=@audio.wav -F response_format=json
```

`-ml 100000` matters: the server otherwise forces 60-character segments and
splits words in half (`fl uttered`). `deploy/serve.sh` sets it for you, and
`deploy/api_check.py` verifies a live service against reference transcripts.

## Speed

**The clean comparison is CPU against CPU** — one machine, one runtime, one
binary, the only difference being the weights. 66 s of audio, 16 threads,
batch 1, Xeon:

| | fp16 | ternary Q2_0 |
|---|---|---|
| file on disk | 465 MiB | **140 MiB** |
| encoder / 30 s window | 578 ms | **473 ms** |
| decode / step | 6.8 ms | **6.0 ms** |

### CPU against H200 — an illustration, not a hardware benchmark

This one compares two things at once: hardware *and* runtime (hand-written
C/AVX2 in whisper.cpp against PyTorch/Transformers). Read it as a statement
about that pair, not about silicon.

| 66 s of audio | H200, PyTorch fp16 | CPU, this model |
|---|---|---|
| encoder / 30 s window | 7.1 ms | 550 ms |
| decode / token | 8.3 ms | **6.0 ms** |
| **end to end (excl. model load)** | **1.17 s** | 2.7 s |

A single decoder step is faster on the CPU — at batch 1 the GPU is launch-bound
(439 kernel launches per decoder pass, ~15 µs each) rather than compute-bound.
But **end to end the H200 still wins by 2.3×**, because the encoder is a dense
matmul over 1500 frames and there the GPU is 77× ahead. For the CPU's 2.3 ms
per-token advantage to repay the encoder's 543 ms deficit you would need about
**236 decoded tokens per 30-second window** — far more than ordinary speech
produces.

Caveats on this pair: the GPU figure excludes mel extraction (~37 ms) which the
CPU figure includes; model load is excluded from both (it is 447 ms fp16 against
950 ms for the ternary file, whose encoder expansion at load is an unresolved
826 ms — see *What is broken*).

## What is broken

Named plainly, because a quantisation result with no failure list has not been
looked at hard enough.

- **No punctuation.** 0.0–0.7 % of utterances contain a comma depending on mode; stock
  whisper-small manages 63.3 %. Capitals appear in 2 %, against 98 %. In timestamp mode a
  sentence-final period survives on 15 % of utterances — that is residue from
  the original model, not learned punctuation. Measured on 300 utterances only,
  [`results/diagnostics/punctuation_mode_switch.json`](results/diagnostics/punctuation_mode_switch.json).
- **English only.** Russian degrades by 12.6×. The fp16 control trained the same
  way degrades identically, so the cause is monolingual fine-tuning, not
  ternarisation.
- **Out of domain**, English is roughly twice as bad as on LibriSpeech.
- **The tied embedding head is not quantised.** It holds 18 % of the parameters
  and 86 of the 140 MiB. A one-shot ternary projection of it, with no dedicated
  QAT, takes WER from 2.71 % to 1152 %. That shows *this* projection fails on
  *that* tensor — it does not show the head is incompressible. A learned Q4 head
  would plausibly bring the file to ~78 MiB and is the most promising next step.
- **The encoder is expanded to Q8_0 in RAM**, so the running model is a hybrid,
  not uniformly 2-bit (see the table at the top).
- **Model load takes 950 ms against 447 ms for fp16.** Three hypotheses — lookup
  table, redundant staging buffer, pointer aliasing — were each tested and each
  wrong: a control `memset` of the same size into the same buffer takes 4 ms
  while the expansion loop takes 826 ms. Unresolved; needs a profiler, not more
  guessing.

## What ternarisation actually costs

Comparing against stock whisper-small answers "is this model good". It does not
answer "what did quantisation cost", because our model was also fine-tuned. So
we ran an fp16 control on the same code path, same data, same schedule, with
quantisation switched off:

- the gap holds at **≈0.37 pp absolute** on test-clean and does not narrow;
- **neither branch reached a floor** — both were still falling almost linearly
  when the runs ended (−0.173 and −0.186 pp per 10k steps);
- so "ternary converges to the same place, just later" remains **unproven**. It
  needs a run to plateau, which we did not do.

## Reproduce

Training is **two stages**, and they are not interchangeable with a single
80k-step run — stage 1 was scheduled for 120 000 steps and stopped at 70 000, so
its cosine schedule differs from a run configured for 70 000.

```bash
pip install -e wal_tat

# 0. data — checked before training, always
python wal_tat/experiments/qat_build_packed.py --split train-clean-100 --out cache/packed30-955h
python wal_tat/experiments/qat_build_repair.py --out cache/repair-ts

# 1. main run: schedule 120k, take the step-70000 checkpoint
python wal_tat/experiments/qat_train.py --model small --precision t3 \
    --data cache/packed30-955h \
    --steps 120000 --batch 64 --lr 3e-4 --warmup 1000 --min-lr-ratio 0.05 \
    --kl-weight 1.0 --ce-weight 1.0 --feature-weight 0.5 --temperature 2.0 \
    --fp-lr-mult 0.25 --scale-lr-mult 0.1 --grad-clip 1.0 --seed 0 \
    --save-every 10000 --out results/qat/small_t3_packed_v2

# 2. repair stage: 10k more, starting FROM that checkpoint, on both corpora
python wal_tat/experiments/qat_train.py --model small --precision t3 \
    --init-from results/qat/small_t3_packed_v2/step070000.pt \
    --data cache/packed30-955h cache/repair-ts \
    --steps 10000 --batch 64 --lr 1e-4 --warmup 200 \
    --ce-weight 1.0 --kl-weight 1.0 --feature-weight 0.5 \
    --fp-lr-mult 0.25 --scale-lr-mult 0.1 \
    --eval-every 2500 --eval-n 2703 --eval-batch-size 16 --max-new-tokens 440 \
    --save-every 2500 --log-every 100 --out results/qat/small_t3_repair_v3

# 3. evaluate under the pinned protocol
bash wal_tat/check_wer.sh

# 4. export for CPU
python deploy/materialize_hf_dense.py            # checkpoint -> dense HF model
python models/convert-h5-to-ggml.py ...          # -> ggml fp16 (whisper.cpp's own script)
python deploy/export_ggml_q2_0.py                # -> 192 matrices as Q2_0 blocks
```

Pinned for reproduction:

| | |
|---|---|
| parent checkpoint | `small_t3_packed_v2/step070000.pt`, sha256 `5672f7aec5b0badf…` |
| final checkpoint | `small_t3_repair_v3/step010000.pt`, sha256 `9687b56ea0f3e6b9…` |
| seed | 0 |
| optimiser | AdamW, β=(0.9, 0.95), ε=1e-8, weight decay 0, grad clip 1.0 |
| schedule | cosine, `min_lr_ratio` 0.05 |
| autocast | bf16 |
| environment | Python 3.13.9, torch 2.12.0+cu130, transformers 5.8.1, CUDA 13.0 |
| whisper.cpp | patch applies on `080bbbe` |

Two rules the project learned the expensive way, both encoded in the scripts:

**Check the dataset before training, every time.** Teacher labels once contained
runaway repetitions — up to 108 repeated 5-grams in a single label. The model
learned them and started looping. Five hours lost to something a one-minute
check catches.

**Never cut audio.** A window ends at an utterance boundary; if the next
utterance does not fit, the window closes. Maximum exactly 30.000 s.

## Method

Ternary projection is not rounding. For each group of 128 inputs we solve for
the codes and scale that minimise the *activation-weighted* error — weights that
multiply large activations are worth more than weights that do not
(`src/wal_tat/scoring.py`). That projection initialises the codes; training then
adjusts them.

Training is quantisation-aware with a straight-through estimator: the forward
pass uses hard ternary codes, the backward pass flows into a latent fp32 weight.
The loss combines KL against a bf16 teacher, cross-entropy on labels, and
feature matching (`src/wal_tat/qat/distill.py`).

A trap worth naming, and it is **specific to this design**. In classical QAT the
high-precision master weights genuinely are the source the deployed checkpoint
is rebuilt from. Here they are not: the group scales are separately learned
parameters, so re-projecting from the latent tensor recomputes scales that were
never trained, and the latent tensor itself is only a surrogate for the gradient
— free to drift outside its quantisation cells. Re-projecting from it instead of
exporting the trained (codes, scales) pair gave **1000.56 % WER**; the latent
weights had drifted to 3.8× the median with 60 % saturated. The deployment
representation is the trained pair, from the start.

## Measurement protocol

Pinned, because it is part of the result and not a convenience:

```
batch_size     = 16
max_new_tokens = 440        (128 truncated output and hid a runaway)
greedy, num_beams = 1
```

**Batch size shifts WER by 0.013 pp on identical code** — batch 16 gives
3.4298 %, batch 8 gives 3.4224 %, batch 1 gives 3.4168 %. This is not random
noise but systematic implementation sensitivity (reduction order, GEMM kernel
selection, floating-point accumulation). So the batch size is pinned, and
differences under 0.02 pp are not treated as signal without a separate paired
test. No confidence intervals are reported anywhere in this repo yet — the
7.3817 % / 7.5002 % gap on test-other in particular has not been shown to
exceed its uncertainty.

`dev-clean` is biased: it drove eval during training, so it reads ~0.04 pp
optimistic. Headline numbers use `test-clean` and `test-other`.

**But those are selection sets, not sealed tests.** They took no part in
gradient training, yet across the project they were measured 21 times over four
runs — and the final checkpoint was chosen over its own step-7500 sibling by
looking at them. That makes them an *evaluation and selection suite*. A
publication-grade claim needs a fresh one-shot set never used for selection,
with the checkpoint hash, protocol and acceptance criteria written down before
the results are opened. That has not been done.

## whisper.cpp: what we had to fix upstream

In the upstream revision I pinned (`080bbbe`), `Q2_0` was declared in ggml but
the whisper quantise path explicitly rejected it and x86 fell back to the
generic scalar kernel — so for Whisper this path was not in a working end-to-end
state. I make no claim about the whole ecosystem's history. Four things were
wrong, and the patch fixes all of them:

1. `Q2_0` was listed as unsupported in the quantise tool — the path was
   unreachable, so nobody had exercised it.
2. **No SIMD kernel on x86.** The scalar fallback made the encoder ~11× slower
   than fp16. `deploy/whisper_cpp_q2_0.patch` adds an AVX2 `vec_dot_q2_0_q8_0`
   (12.5 s → 2.4 s per encoder window).
3. The loader forced the tied embedding to the model's weight type; at 2 bits
   that is catastrophic, so it now stays fp16.
4. Encoder matrices are promoted `Q2_0 → Q8_0` at load. The promotion is exact —
   ternary codes and the same fp16 scale are representable in Q8_0 — and buys
   Q8_0 kernel speed on the compute-bound encoder while the decoder keeps the
   2-bit bandwidth win.

`deploy/test_q2_0.c` checks the kernel against an exact float reference.

## Layout

```
wal_tat/src/wal_tat/     the library — quantisers, QAT, projection, packing, kernels
wal_tat/experiments/     data building and training entry points
wal_tat/check_wer.py     evaluation under the pinned protocol
deploy/                  exporter, whisper.cpp patch, kernel test, service scripts
results/diagnostics/     every number in this README, traceable to its measurement
PROJECT_RULES_RU.md      the working rules this project ran under (Russian)
```

## Licence

MIT. The whisper.cpp patch is offered upstream under that project's licence.

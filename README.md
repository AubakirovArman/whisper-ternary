# Ternary Whisper

**whisper-small with 192 weight matrices in {−1, 0, +1}, and it beats the model it came from.**

| LibriSpeech (greedy, batch 16) | this model | stock whisper-small |
|---|---|---|
| test-clean | **2.7145 %** | 3.3391 % |
| test-other | **7.3817 %** | 7.5002 % |
| long-form (9 recordings, standard HF pipeline) | **3.632 %** | 3.891 % |

Both test sets are genuinely held out — neither took part in training.

198 180 864 weights (81.98 % of the model) carry no magnitude at all: each is
one of three values, with an fp16 scale shared across every 128 inputs. That is
2.125 bits per weight. The file is **140 MiB against 465 MiB** for the fp16
original, and it runs on a CPU through a patched whisper.cpp.

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

## Speed, honestly

Same weights, same 66 s of audio, 16 threads, batch 1, on a Xeon:

| | fp16 | ternary Q2_0 |
|---|---|---|
| file on disk | 465 MiB | **140 MiB** |
| encoder / 30 s window | 578 ms | **473 ms** |
| decode / step | 6.8 ms | **6.0 ms** |

And against an H200 running the same weights in PyTorch:

| | H200 (PyTorch fp16) | CPU (this model) |
|---|---|---|
| encoder / window | 7.1 ms | 550 ms |
| **decode / token** | 8.3 ms | **6.0 ms** |

The decoder is faster on the CPU. That is not a typo and it is not a triumph of
the CPU — at batch 1 the GPU spends its time launching kernels (439 per decoder
pass, ~15 µs each) rather than computing. The encoder, a dense matmul over 1500
frames, is exactly what a GPU is for, and there the H200 wins by 77×.

## What is broken

Named plainly, because a quantisation result with no failure list has not been
looked at hard enough.

- **No punctuation.** 0.0–0.7 % of utterances contain a comma depending on mode; stock
  whisper-small manages 63.3 %. Capitals appear in 2 %, against 98 %. In timestamp mode a
  sentence-final period survives on 15 % of utterances — that is residue from
  the original model, not learned punctuation. Measured on 300 utterances,
  [`results/diagnostics/punctuation_mode_switch.json`](results/diagnostics/punctuation_mode_switch.json).
- **English only.** Russian degrades by 12.6×. The fp16 control trained the same
  way degrades identically, so the cause is monolingual fine-tuning, not
  ternarisation.
- **Out of domain**, English is roughly twice as bad as on LibriSpeech.
- **The tied embedding head is not quantised.** It holds 18 % of the parameters
  and 86 of the 140 MiB. Forcing it to 2 bits moves WER from 2.71 % to 1152 %.

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

```bash
pip install -e wal_tat

# 1. data — checked before training, always
python wal_tat/experiments/qat_build_packed.py --split train-clean-100 --out cache/packed30
python wal_tat/experiments/qat_build_repair.py --out cache/repair-ts

# 2. quantisation-aware training, 80k steps
python wal_tat/experiments/qat_train.py --model small --precision t3 \
    --data cache/packed30 cache/repair-ts --steps 70000 --batch 64

# 3. evaluate under the pinned protocol
bash wal_tat/check_wer.sh

# 4. export for CPU
python deploy/materialize_hf_dense.py            # checkpoint -> dense HF model
python models/convert-h5-to-ggml.py ...          # -> ggml fp16 (whisper.cpp's own script)
python deploy/export_ggml_q2_0.py                # -> 192 matrices as Q2_0 blocks
```

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

A trap worth naming: **the latent weight is not a master copy.** Re-projecting
from latent weights instead of exporting the trained (codes, scales) pair gave
1000.56 % WER — the latent weights drift by 3.8× the median and 60 % saturate.
The master is the pair, always.

## Measurement protocol

Pinned, because it is part of the result and not a convenience:

```
batch_size     = 16
max_new_tokens = 440        (128 truncated output and hid a runaway)
greedy, num_beams = 1
```

**Harness noise floor is ±0.013 pp** — batch-16 gives 3.4298 %, batch-8 3.4224 %,
batch-1 3.4168 % on identical code. Differences below 0.02 pp are not signal.

`dev-clean` is biased: it drove eval during training and decisions were made on
it, so it reads ~0.04 pp optimistic. Headline numbers use `test-clean` and
`test-other`.

## whisper.cpp: what we had to fix upstream

The `Q2_0` type existed in ggml but had never been run end to end. Four things
were wrong, and the patch fixes all of them:

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

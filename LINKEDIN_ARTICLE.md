# I removed the magnitude from 198 million weights. The model got more accurate.

Most of a neural network's weights don't need to be numbers. I spent the last
stretch proving that on a speech model, and the result surprised me more than it
probably should have.

**whisper-small, with 192 of its weight matrices reduced to three possible values
— −1, 0, and +1 — is more accurate than the fp16 model it was distilled from.**

| LibriSpeech, greedy decoding | ternary | stock whisper-small |
|---|---|---|
| test-clean | **2.71 %** WER | 3.34 % |
| test-other | **7.38 %** WER | 7.50 % |
| long-form, 9 recordings | **3.63 %** | 3.89 % |

Both test sets are genuinely held out. 198,180,864 weights — 82 % of the model —
carry no magnitude at all. Each is one of three values, with a single fp16 scale
shared across every 128 inputs. That works out to **2.125 bits per weight**. The
deployable file is 140 MiB instead of 465, and it runs on a laptop CPU.

Weights and code are public, and every number above is traceable to a
measurement file in the repo. Links at the bottom.

---

## The part I'd tell another engineer over coffee

Three things came out of this that I didn't expect, and they're the reason I
think the work was worth doing.

### 1. The decoder is faster on a CPU than on an H200

Same weights, same audio, batch size 1:

| | H200, PyTorch fp16 | CPU, ternary |
|---|---|---|
| decode / token | 8.3 ms | **6.0 ms** |
| encoder / 30 s window | 7.1 ms | 550 ms |

That's not a story about CPUs being secretly great. It's a story about what
autoregressive decoding actually is at batch 1: **439 kernel launches per decoder
pass, roughly 15 µs each, regardless of matrix size.** The GPU spends its time
in launch overhead, not arithmetic. A thirty-thousand-dollar accelerator idles
while the CPU, which has no launch overhead to pay, quietly wins.

The encoder tells the opposite story — a dense matmul over 1500 frames is
precisely what a GPU is for, and there the H200 wins by 77×.

The lesson generalises past this model: **before optimising a kernel, find out
whether you're compute-bound, memory-bound, or launch-bound.** I measured a
"1.48× speedup" early on that turned out to be me comparing CUDA graphs against
no CUDA graphs. Different question entirely.

### 2. One tensor can hold the whole model hostage

Whisper ties its decoder's token embedding to its output projection. It's 18 %
of the parameters and, in the deployed file, 86 of 140 MiB — by far the largest
single thing left in fp16. So I quantised it too.

**WER went from 2.71 % to 1152 %.**

Not degraded — destroyed. 563,054 spurious insertions. That tensor was never
trained under quantisation and cannot survive two bits, and now I have the
number to say so instead of a hunch. It also meant vanilla whisper.cpp
compatibility was impossible: its loader assigns the model's weight type to
every 2D tensor, including that one. The fix is one line, and it's in the patch.

### 3. A two-bit format shipped in ggml that nobody had ever run

whisper.cpp and ggml already had a `Q2_0` type — 2.25 bits, group of 64, exactly
the right container for ternary codes. Perfect fit. Except:

- it was listed as **unsupported** in the quantisation tool, so the path was
  unreachable;
- there was **no SIMD kernel on x86** — the scalar fallback made the encoder 11×
  slower than fp16;
- quantising the original model with it produced pure garbage.

The type had been declared and never exercised end to end. I wrote the missing
AVX2 kernel (12.5 s → 2.4 s per encoder window), fixed the two gates, and added
an exact `Q2_0 → Q8_0` promotion for the compute-bound encoder so it runs at
Q8_0 speed while the decoder keeps the 2-bit bandwidth win. That patch is
offered upstream.

---

## What doesn't work

A compression result without a failure list hasn't been looked at hard enough.
Mine:

- **No punctuation.** 0.0 % of utterances contain a comma. Stock whisper-small
  manages 63.3 %. Capitals: 2 % against 98 %. The cause is my training labels,
  which were lowercased and unpunctuated — the fix is data I've already built
  and haven't yet run.
- **English only.** Russian degrades 12.6×. But an fp16 control trained the same
  way degrades identically, so the cause is monolingual fine-tuning, **not**
  ternarisation. Those are different accusations and I'm careful not to swap
  them.
- **The encoder is still 2× slower than fp16** on CPU in wall-clock terms, even
  after the kernel work.

And the honest version of the headline: beating stock whisper-small answers "is
this model good", not "what did quantisation cost" — my model was also
fine-tuned. So I ran an fp16 control on the same code path, same data, same
schedule, with quantisation switched off. **The gap holds at ~0.37 pp and does
not narrow. Neither branch reached a floor.** Whether ternary eventually
converges to the same place is unproven, and I say so rather than implying it.

---

## The mistake worth publishing

At one point I re-projected the ternary codes from the latent fp32 weights that
QAT maintains during training. Reasonable-sounding: the latent weight is the
"real" weight, the codes are a shadow of it.

**Result: 1000.56 % WER.**

The latent weights had drifted to 3.8× the median and 60 % were saturated. They
are a training scratchpad, not a master copy. **The master is the pair (codes,
scales)** — the thing that was actually trained. I now believe this failure mode
catches most people who implement QAT and export it themselves, and I've written
it into the repo so the next person gets it for free.

---

## Try it

```bash
hf download armanibadboy/whisper-small-ternary ggml-small-wal-ternary-q2_0.bin
./whisper-cli -m ggml-small-wal-ternary-q2_0.bin -f audio.wav -ng
```

Every quantised matrix holds at most three distinct values per group of 128 —
the repo shows you how to check that yourself in four lines of PyTorch rather
than taking my word for it.

- **Model:** huggingface.co/armanibadboy/whisper-small-ternary
- **Code, recipe, and all measurement files:** github.com/AubakirovArman/whisper-ternary

There is remarkably little public information on making speech models ternary
and having them still work. That's most of why I published the whole thing —
the failures, the noise floor, the negative results and the fp16 control —
rather than just the number that looks good.

*If you're working on low-bit quantisation, ggml kernels, or on-device ASR, I'd
genuinely like to compare notes.*

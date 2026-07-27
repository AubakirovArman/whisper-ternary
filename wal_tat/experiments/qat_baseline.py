"""Measure and freeze a full-precision Whisper WER baseline.

Every ternarisation experiment is scored against one of these files, so the
output is written once and then treated as immutable: re-running with an
existing ``--output`` fails unless ``--overwrite`` is passed.

Example
-------
    CUDA_VISIBLE_DEVICES=2 python experiments/qat_baseline.py \
        --model openai/whisper-small --dtype bf16 --split dev-clean \
        --output results/qat/whisper-small_bf16_dev-clean.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from wal_tat import atomic_write_json
from wal_tat.qat import evaluate_wer, load_librispeech

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="HF id or local snapshot path")
    parser.add_argument("--split", default="dev-clean")
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Evaluate the first N utterances (default: the whole split).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--language", default="en")
    parser.add_argument("--task", default="transcribe")
    parser.add_argument(
        "--no-nll",
        action="store_true",
        help="Skip the teacher-forced pass (WER only, roughly 2x faster).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--expect-wer",
        type=float,
        default=None,
        help="Known WER fraction to cross-check against; reported, never enforced.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    print(message, flush=True)


def main() -> None:
    args = parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite frozen baseline {output}")

    dtype = DTYPES[args.dtype]
    device = torch.device(args.device)

    _log(f"[load] dataset {args.split}")
    started = time.perf_counter()
    split = load_librispeech(args.split, n=args.n)
    _log(
        f"[load] {len(split)} utterances, {split.audio_seconds / 3600:.3f} h audio "
        f"in {time.perf_counter() - started:.1f}s "
        f"(ids sha256 {split.sample_ids_sha256[:16]}...)"
    )

    _log(f"[load] model {args.model} ({args.dtype})")
    processor = WhisperProcessor.from_pretrained(
        args.model, local_files_only=args.local_files_only
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
        dtype=dtype,
    )
    model.to(device=device, dtype=dtype).eval()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def progress(state) -> None:
        if state["decoded"] % (args.batch_size * 20) == 0:
            rate = state["decoded"] / max(state["elapsed_seconds"], 1e-9)
            _log(
                f"[decode] {state['decoded']}/{state['total']} "
                f"({rate:.2f} utt/s)"
            )

    result = evaluate_wer(
        model,
        processor,
        split,
        batch_size=args.batch_size,
        language=args.language,
        task=args.task,
        max_new_tokens=args.max_new_tokens,
        compute_nll=not args.no_nll,
        progress=progress,
    )

    payload = {
        "schema_version": 1,
        "kind": "wal-tat-qat-baseline",
        "model": args.model,
        "model_revision": Path(args.model).name,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "expected_wer": args.expect_wer,
        "wer_delta_vs_expected": (
            None if args.expect_wer is None else result["wer"] - args.expect_wer
        ),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "transformers": __import__("transformers").__version__,
            "gpu": (
                torch.cuda.get_device_name(device) if device.type == "cuda" else None
            ),
        },
        "command": " ".join(sys.argv),
        **result,
    }
    atomic_write_json(output, payload)

    summary = {key: payload[key] for key in ("model", "dataset", "samples", "wer")}
    summary["wer_percent"] = 100.0 * payload["wer"]
    summary.update(
        {key: payload[key] for key in ("S", "D", "I", "N") if key in payload}
    )
    if args.expect_wer is not None:
        summary["expected_wer_percent"] = 100.0 * args.expect_wer
        summary["delta_percent"] = 100.0 * payload["wer_delta_vs_expected"]
    if "nll" in payload:
        summary["nll"] = payload["nll"]
    _log(json.dumps(summary, indent=2))
    _log(f"[write] {output}")
    _log("[DONE]")


if __name__ == "__main__":
    main()

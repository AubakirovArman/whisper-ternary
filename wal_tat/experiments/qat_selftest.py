"""Self-test for the QAT quantization core and model surgery.

Checks, in order:

a. ternary round trip on a random ``[768, 768]`` matrix -- exactly three
   distinct values per group, sane relative Frobenius error, and a shuffled
   control that is far worse;
b. conversion of a real Whisper checkpoint -- matrix/weight counts, coverage,
   bits per weight, and a working forward pass;
c. straight-through gradient flow into the latent weight;
d. code churn after a single optimizer step;
e. export/checkpoint round trips.

Run with ``CUDA_VISIBLE_DEVICES=2 python experiments/qat_selftest.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wal_tat.packing import read_packed_matrix, unpack_partial_matrix  # noqa: E402
from wal_tat.qat.convert import (  # noqa: E402
    collect_input_second_moments,
    convert_model_to_qat,
    export_packed_artifact,
    load_qat_checkpoint,
    qat_modules,
    save_qat_checkpoint,
)
from wal_tat.qat.quant import QuantLinear, get_quantizer, grouped_view  # noqa: E402


def banner(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}", flush=True)


def check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}", flush=True)
    if not condition:
        raise AssertionError(message)


# --------------------------------------------------------------------- part a


def test_round_trip(device: torch.device, group_size: int = 128) -> None:
    banner("(a) ternary round trip on a random [768, 768] weight")
    generator = torch.Generator(device="cpu").manual_seed(20260725)
    weight = torch.randn(768, 768, generator=generator) * 0.02

    def build(learnable: bool) -> QuantLinear:
        module = QuantLinear(
            768,
            768,
            bias=False,
            precision="t3",
            group_size=group_size,
            learnable_scales=learnable,
            device=device,
        )
        with torch.no_grad():
            module.weight.copy_(weight.to(device))
        return module

    # Derived scales are recomputed from the latent weight on every call, so
    # this measures the plain BitNet absmean round trip.
    layer = build(learnable=False)
    codes, scales = layer.export_codes()
    check(tuple(codes.shape) == (768, 6, 128), f"codes shape {tuple(codes.shape)}")
    check(tuple(scales.shape) == (768, 6), f"scales shape {tuple(scales.shape)}")
    check(codes.dtype == torch.int8, f"codes dtype {codes.dtype}")
    check(scales.dtype == torch.float16, f"scales dtype {scales.dtype}")

    unique = torch.unique(codes)
    check(
        set(unique.tolist()) <= {-1, 0, 1},
        f"global code alphabet {sorted(unique.tolist())}",
    )
    approx = codes.float() * scales.float().unsqueeze(-1)
    per_group_levels = torch.tensor(
        [
            torch.unique(row).numel()
            for row in approx.reshape(-1, group_size)
        ]
    )
    check(
        int(per_group_levels.max().item()) <= 3,
        f"max distinct dequantized values in any group = "
        f"{int(per_group_levels.max().item())} (<= 3)",
    )
    print(
        f"  distinct values per group: min={int(per_group_levels.min())} "
        f"median={int(per_group_levels.median())} max={int(per_group_levels.max())}"
    )

    grouped = grouped_view(layer.weight.detach(), group_size)
    total = grouped.square().sum()
    matched = ((grouped - approx).square().sum() / total).sqrt().item()

    # Shuffled control: same codes, permuted inside every group.
    permutation = torch.argsort(
        torch.rand(approx.shape, device=approx.device), dim=-1
    )
    shuffled = torch.gather(approx, -1, permutation)
    control = ((grouped - shuffled).square().sum() / total).sqrt().item()

    stats = layer.quant_error()
    print(f"  relative Frobenius error (absmean-derived init): {matched:.4f}")
    print(f"  shuffled control                              : {control:.4f}")
    print(f"  quant_error(): {json.dumps(stats, indent=4)}")
    check(matched < 0.7, f"relative Frobenius error {matched:.4f} < 0.70")
    check(
        control > 1.0,
        f"shuffled control {control:.4f} > 1.0 (worse than emitting zeros)",
    )
    check(
        control > 2 * matched,
        f"shuffled control {control:.4f} > 2x matched {matched:.4f}",
    )

    # Learned scales initialized from the unweighted optimal projection must
    # not be worse than absmean.
    learned = build(learnable=True)
    learned.reset_scales_from_projection()
    unweighted = learned.quant_error()["relative_frobenius"]
    print(f"  relative Frobenius error (optimal projection)  : {unweighted:.4f}")
    check(
        unweighted <= matched + 1e-6,
        f"projection init {unweighted:.4f} <= absmean {matched:.4f}",
    )

    # The exact activation-weighted projection must beat absmean under its own
    # metric and must be a fixed point of the encoder: re-encoding the latent
    # weight against the projected scale reproduces the projected codes.
    moment = (torch.rand(768, device=device) * 4.0 + 0.1).float()
    projection_stats = learned.reset_scales_from_projection(moment)
    projected_codes, _, _ = get_quantizer("t3").project(
        learned.weight.detach(), moment, group_size=group_size
    )
    exported_codes, _ = learned.export_codes()
    disagreements = int((exported_codes != projected_codes).sum().item())
    fraction = disagreements / projected_codes.numel()
    check(
        fraction < 1e-3,
        f"encoder reproduces projected codes ({disagreements} of "
        f"{projected_codes.numel()} differ, {fraction:.2e}; residual is FP16 "
        "scale rounding at the +/- scale/2 decision boundary)",
    )
    weighted_before = layer.quant_error(moment)["weighted_relative_error"]
    weighted_after = learned.quant_error(moment)["weighted_relative_error"]
    print(
        f"  weighted relative error: absmean={weighted_before:.4f} -> "
        f"activation-weighted projection={weighted_after:.4f} "
        f"(projection reported {projection_stats['weighted_relative_error']:.4f})"
    )
    check(
        weighted_after < weighted_before,
        "activation-weighted projection init improves the weighted error",
    )

    # Straight-through identity: the quantized forward is exactly codes*scale.
    with torch.no_grad():
        check(
            torch.equal(learned.quantized_weight(), learned.dequantized_weight()),
            "quantized_weight() executes the exported codes exactly",
        )

    # Binary sanity.
    binary = QuantLinear(
        768,
        768,
        bias=False,
        precision="b1",
        group_size=group_size,
        learnable_scales=False,
        device=device,
    )
    with torch.no_grad():
        binary.weight.copy_(weight.to(device))
    binary_codes, _ = binary.export_codes()
    check(
        set(torch.unique(binary_codes).tolist()) <= {-1, 1},
        "binary alphabet is {-1, +1}",
    )
    print(f"  binary relative Frobenius error: "
          f"{binary.quant_error()['relative_frobenius']:.4f}")


# --------------------------------------------------------------------- part b


def load_whisper(model_id: str, device: torch.device, dtype: torch.dtype):
    from transformers import WhisperForConditionalGeneration

    model = WhisperForConditionalGeneration.from_pretrained(model_id, dtype=dtype)
    return model.to(device).eval()


def whisper_batch(model, device: torch.device, dtype: torch.dtype, batch: int = 2):
    config = model.config
    features = torch.randn(
        batch, config.num_mel_bins, 2 * config.max_source_positions,
        device=device, dtype=dtype,
    )
    labels = torch.randint(
        0, config.vocab_size, (batch, 12), device=device, dtype=torch.long
    )
    return features, labels


def test_conversion(model_id: str, device: torch.device, expected: dict) -> dict:
    banner(f"(b) convert {model_id}")
    dtype = torch.bfloat16
    model = load_whisper(model_id, device, dtype)
    baseline_parameters = sum(p.numel() for p in model.parameters())
    features, labels = whisper_batch(model, device, dtype)

    with torch.no_grad():
        reference = model(input_features=features, labels=labels).logits.float()

    moments = collect_input_second_moments(
        model, lambda: model(input_features=features, labels=labels)
    )
    print(f"  calibration moments collected for {len(moments)} modules")

    manifest = convert_model_to_qat(
        model,
        precision="t3",
        group_size=128,
        input_second_moments=moments,
    )
    print(f"  {manifest.summary()}")
    check(
        manifest.num_matrices == expected["matrices"],
        f"matrices = {manifest.num_matrices} (expected {expected['matrices']})",
    )
    check(
        manifest.quantized_weights == expected["weights"],
        f"quantized weights = {manifest.quantized_weights:,} "
        f"(expected {expected['weights']:,})",
    )
    check(
        manifest.model_parameters == baseline_parameters,
        f"parameter count unchanged at {manifest.model_parameters:,}",
    )
    check(
        len(qat_modules(model)) == expected["matrices"],
        "every target module is a QuantLinear",
    )
    check(
        all(
            not isinstance(module, nn.Linear)
            for name, module in model.named_modules()
            if name.rsplit(".", 1)[-1]
            in {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}
        ),
        "no target nn.Linear survived the swap",
    )
    check(
        manifest.untouched_linear == ("proj_out",),
        f"untouched linear layers: {manifest.untouched_linear}",
    )
    check(
        abs(manifest.matrix_bpw - 2.125) < 1e-6,
        f"matrix bpw = {manifest.matrix_bpw:.4f} (2 code bits + 16/128 scale bits)",
    )
    print(
        f"  coverage: {100 * manifest.linear_coverage:.2f}% of linear weights, "
        f"{100 * manifest.parameter_coverage:.2f}% of all parameters"
    )
    print(
        f"  size: {manifest.total_bits / 8 / 2**20:.1f} MiB vs "
        f"{manifest.model_parameters * 2 / 2**20:.1f} MiB at bf16 "
        f"-> {manifest.compression_ratio:.2f}x"
    )

    with torch.no_grad():
        quantized_logits = model(input_features=features, labels=labels).logits.float()
    check(torch.isfinite(quantized_logits).all(), "quantized forward pass is finite")
    check(
        quantized_logits.shape == reference.shape,
        f"logits shape preserved {tuple(quantized_logits.shape)}",
    )
    drift = (quantized_logits - reference).norm() / reference.norm()
    print(f"  relative logit drift after conversion (untrained): {drift:.4f}")
    check(drift > 0, "conversion actually changed the weights")
    return {"model": model, "manifest": manifest, "batch": (features, labels)}


# ------------------------------------------------------------------ parts c, d


def test_gradients_and_flips(state: dict, device: torch.device) -> None:
    banner("(c) straight-through gradient flow + (d) code churn after a step")
    model = state["model"]
    features, labels = state["batch"]
    modules = qat_modules(model)
    probe_names = [
        "model.encoder.layers.0.self_attn.q_proj",
        "model.decoder.layers.0.fc1",
    ]
    before = {name: modules[name].export_codes()[0].clone() for name in probe_names}

    model.train()
    for parameter in model.parameters():
        parameter.requires_grad_(parameter.is_floating_point())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=5e-4)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        loss = model(input_features=features, labels=labels).loss
    loss.backward()
    print(f"  teacher-forced CE loss: {loss.item():.4f}")

    for name in probe_names:
        module = modules[name]
        check(module.weight.grad is not None, f"{name}: latent weight has a gradient")
        grad_norm = module.weight.grad.norm().item()
        nonzero = (module.weight.grad != 0).float().mean().item()
        check(grad_norm > 0, f"{name}: |grad| = {grad_norm:.3e} > 0")
        check(
            nonzero > 0.5,
            f"{name}: {100 * nonzero:.1f}% of latent entries have a non-zero grad",
        )
        check(
            module.group_scale.grad is not None
            and module.group_scale.grad.norm().item() > 0,
            f"{name}: group scale gradient |g| = "
            f"{module.group_scale.grad.norm().item():.3e}",
        )
        zero_codes = (before[name] == 0)
        if zero_codes.any():
            padded = torch.nn.functional.pad(
                module.weight.grad, (0, module.padding)
            ).reshape(module.out_features, module.groups, module.group_size)
            check(
                (padded[zero_codes] != 0).float().mean().item() > 0.5,
                f"{name}: gradient reaches weights whose current code is 0 "
                "(codes can flip out of the dead zone)",
            )

    torch.nn.utils.clip_grad_norm_(parameters, 1.0)
    optimizer.step()
    for module in modules.values():
        module.constrain_()

    total_flips = 0
    total_codes = 0
    for name in probe_names:
        module = modules[name]
        flips = module.code_flips(before[name])
        count = before[name].numel()
        total_flips += flips
        total_codes += count
        print(
            f"  {name}: {flips:,} / {count:,} codes flipped "
            f"({100 * flips / count:.3f}%)"
        )
        check(flips > 0, f"{name}: optimizer step changed the discrete codes")
    print(f"  probed total: {total_flips:,} / {total_codes:,} codes flipped")
    model.eval()


# --------------------------------------------------------------------- part e


def test_persistence(state: dict, device: torch.device) -> None:
    banner("(e) checkpoint and packed-artifact round trips")
    model = state["model"]
    manifest = state["manifest"]
    features, labels = state["batch"]
    modules = qat_modules(model)
    with torch.no_grad():
        expected_logits = model(input_features=features, labels=labels).logits.float()

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = save_qat_checkpoint(
            model, root / "qat.pt", manifest=manifest, extra={"step": 1}
        )
        size_mib = path.stat().st_size / 2**20
        print(f"  checkpoint written: {size_mib:.1f} MiB")

        fresh = load_whisper(state["model_id"], device, torch.bfloat16)
        restored, extra = load_qat_checkpoint(fresh, path, map_location=str(device))
        check(extra == {"step": 1}, f"extra payload round trip {extra}")
        check(
            restored.quantized_weights == manifest.quantized_weights,
            "restored manifest matches",
        )
        with torch.no_grad():
            restored_logits = fresh(
                input_features=features, labels=labels
            ).logits.float()
        check(
            torch.equal(restored_logits, expected_logits),
            "restored model reproduces logits exactly",
        )

        name = "model.encoder.layers.0.self_attn.q_proj"
        summary = export_packed_artifact(model, root / "packed", manifest=manifest)
        print(
            f"  packed artifact: {summary['matrices']} files, "
            f"{summary['artifact_bytes'] / 2**20:.2f} MiB, "
            f"first-file bpw {summary['files'][0]['true_bpw']:.4f}"
        )
        packed = read_packed_matrix(root / "packed" / f"{name}.waltq2")
        restored_weight = unpack_partial_matrix(packed).to(device)
        deployed = modules[name].dequantized_weight()
        check(
            torch.equal(restored_weight, deployed),
            "packed artifact reproduces the deployed matrix bit-exactly",
        )
        artifact_bpw = summary["artifact_bytes"] * 8 / manifest.quantized_weights
        print(f"  serialized artifact bpw over quantized weights: {artifact_bpw:.4f}")
        check(artifact_bpw < 2.2, f"serialized bpw {artifact_bpw:.4f} < 2.2")
        del fresh


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["openai/whisper-tiny", "openai/whisper-small"],
    )
    arguments = parser.parse_args()

    expectations = {
        "openai/whisper-tiny": {"matrices": 64, "weights": 16_515_072},
        "openai/whisper-small": {"matrices": 192, "weights": 198_180_864},
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    print(f"torch: {torch.__version__}")
    torch.manual_seed(0)

    test_round_trip(device)
    for model_id in arguments.models:
        state = test_conversion(model_id, device, expectations[model_id])
        state["model_id"] = model_id
        test_gradients_and_flips(state, device)
        test_persistence(state, device)
        del state
        torch.cuda.empty_cache()

    banner("all self-tests passed")
    print("[DONE]", flush=True)


if __name__ == "__main__":
    main()

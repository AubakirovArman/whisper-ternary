from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

import build_audit_holdout as audit_builder  # noqa: E402
from build_audit_holdout import (  # noqa: E402
    c4_arrow_pattern,
    text_windows,
    wikitext103_arrow_pattern,
)


def test_text_windows_uses_declared_offset_and_extra_label_token() -> None:
    ids = torch.arange(20)
    windows = text_windows(ids, count=2, length=4, offset=3)
    assert [value.tolist() for value in windows] == [
        [3, 4, 5, 6, 7],
        [7, 8, 9, 10, 11],
    ]


def test_text_windows_rejects_short_stream() -> None:
    with pytest.raises(RuntimeError):
        text_windows(torch.arange(5), count=2, length=4, offset=0)


def test_c4_train_shard_pattern_is_explicit_and_zero_padded() -> None:
    assert "c4-train-00001-of-" in c4_arrow_pattern("train", 1)
    assert c4_arrow_pattern("validation", 999).endswith("c4-validation.arrow")
    with pytest.raises(ValueError, match="non-negative"):
        c4_arrow_pattern("train", -1)


def test_wikitext103_pattern_selects_one_explicit_shard() -> None:
    assert "wikitext-train-00001-of-" in wikitext103_arrow_pattern("train", 1)
    assert wikitext103_arrow_pattern("validation", 999).endswith(
        "wikitext-validation.arrow"
    )
    with pytest.raises(ValueError, match="0 or 1"):
        wikitext103_arrow_pattern("train", 2)


def test_optional_code_source_is_resolved_without_a_hard_import(monkeypatch) -> None:
    monkeypatch.setattr(
        audit_builder.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=f"/tmp/{name}/__init__.py"),
    )
    assert audit_builder.code_source_directories("scipy") == (Path("/tmp/scipy"),)

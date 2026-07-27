import sys
from pathlib import Path


EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))

from recovery_suite_overlap_audit import (  # noqa: E402
    intervals_overlap,
    source_ranges,
    stream_identity,
)


def test_source_ranges_prefers_full_stream_identity():
    payload = {
        "full_token_stream_sha256": {"squad": "full"},
        "token_sha256": {"squad": "selected-windows"},
        "ranges": {"calibration": {"squad": [100, 200]}},
    }
    result = source_ranges(payload)
    assert set(result) == {"full"}
    assert result["full"][0]["start"] == 100


def test_adjacent_ranges_do_not_overlap_but_intersecting_ranges_do():
    assert not intervals_overlap({"start": 0, "end": 10}, {"start": 10, "end": 20})
    assert intervals_overlap({"start": 0, "end": 11}, {"start": 10, "end": 20})


def test_legacy_audit_offsets_use_source_file_identity():
    payload = {
        "format": "wal-tat-audit-holdout-v1",
        "model_revision": "revision",
        "sequence_length": 4,
        "offsets": {"squad_context": 20, "code": 40},
        "gates": {"squad_context": [0, 1], "torch_code": [0]},
        "sources": ["/cache/squad-validation.arrow", "/lib/a.py"],
        "source_sha256": {
            "/cache/squad-validation.arrow": "squad-sha",
            "/lib/a.py": "code-sha",
        },
    }
    ranges = source_ranges(payload)
    squad_id = stream_identity(payload, "squad_context")
    code_id = stream_identity(payload, "torch_code")
    assert ranges[squad_id][0]["start"] == 20
    assert ranges[squad_id][0]["end"] == 28
    assert ranges[code_id][0]["start"] == 40
    assert ranges[code_id][0]["end"] == 44


def test_train_and_validation_arrow_sources_have_distinct_identities():
    common = {
        "model_revision": "revision",
        "sequence_length": 4,
        "ranges": {"gates": {"squad_context": [0, 4]}},
    }
    validation = {
        **common,
        "sources": ["/cache/squad-validation.arrow"],
        "source_sha256": {"/cache/squad-validation.arrow": "validation-sha"},
    }
    train = {
        **common,
        "ranges": {"gates": {"squad_train_context": [0, 4]}},
        "sources": ["/cache/squad-train.arrow"],
        "source_sha256": {"/cache/squad-train.arrow": "train-sha"},
    }
    assert stream_identity(validation, "squad_context") != stream_identity(
        train, "squad_train_context"
    )
    assert source_ranges(validation).keys() != source_ranges(train).keys()


def test_c4_text_slices_have_distinct_stream_identities():
    common = {
        "model_revision": "revision",
        "sources": ["/cache/c4-train-00001-of-00002.arrow"],
        "source_sha256": {"/cache/c4-train-00001-of-00002.arrow": "c4-sha"},
    }
    first = {**common, "c4_text_start": 0, "c4_text_count": 4000}
    second = {**common, "c4_text_start": 4000, "c4_text_count": 4000}
    assert stream_identity(first, "c4_train") != stream_identity(second, "c4_train")


def test_wikitext_is_not_classified_as_code():
    payload = {
        "model_revision": "revision",
        "sources": [
            "/cache/wikitext-train-00000-of-00002.arrow",
            "/lib/a.py",
        ],
        "source_sha256": {
            "/cache/wikitext-train-00000-of-00002.arrow": "wiki-sha",
            "/lib/a.py": "code-sha",
        },
        "prose_text_start": 100,
        "prose_text_count": 200,
    }
    assert stream_identity(payload, "wikitext103_train") != stream_identity(
        payload, "datasets_code"
    )


def test_segmented_ranges_are_expanded_for_overlap_checks():
    payload = {
        "full_token_stream_sha256": {"squad": "full"},
        "ranges": {"calibration": {"squad": [[100, 200], [500, 600]]}},
    }
    ranges = source_ranges(payload)["full"]
    assert [(item["start"], item["end"]) for item in ranges] == [
        (100, 200),
        (500, 600),
    ]

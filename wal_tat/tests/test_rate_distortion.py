import pytest

from wal_tat.rate_distortion import (
    FormatOption,
    allocate_rate_distortion,
    groupwise_payload_bits,
)


def option(unit, format, bits, distortion):
    return FormatOption(
        unit=unit,
        format=format,
        payload_bits=bits,
        distortion=distortion,
    )


def test_allocator_takes_best_distortion_reduction_per_bit():
    result = allocate_rate_distortion(
        [
            option("a", "t3", 2, 10.0),
            option("a", "q4", 4, 2.0),
            option("b", "t3", 2, 8.0),
            option("b", "q4", 4, 6.0),
        ],
        target_payload_bits=6,
    )
    assert result.selected["a"].format == "q4"
    assert result.selected["b"].format == "t3"
    assert result.total_payload_bits == 6
    assert result.total_distortion == pytest.approx(10.0)


def test_allocator_removes_dominated_options():
    result = allocate_rate_distortion(
        [
            option("a", "bad", 4, 12.0),
            option("a", "t3", 2, 10.0),
            option("a", "q4", 5, 1.0),
        ],
        target_payload_bits=4,
    )
    assert result.selected["a"].format == "t3"
    assert result.unused_payload_bits == 2


def test_allocator_rejects_budget_below_minimum():
    with pytest.raises(ValueError, match="minimum representation"):
        allocate_rate_distortion(
            [option("a", "t3", 3, 1.0)],
            target_payload_bits=2,
        )


def test_format_option_rejects_invalid_distortion():
    with pytest.raises(ValueError, match="distortion"):
        option("a", "bad", 2, float("nan"))


def test_groupwise_payload_counts_padding_and_scales():
    assert groupwise_payload_bits(
        rows=2,
        columns=130,
        group_size=128,
        code_bits=2,
    ) == 2 * 2 * (128 * 2 + 16)

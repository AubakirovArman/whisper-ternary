from whisper_sharded_full_audit import _windows


def test_windows_preserve_original_batch_membership() -> None:
    windows = _windows(2703, 16, 4)
    assert windows == [
        (0, 688),
        (688, 672),
        (1360, 672),
        (2032, 671),
    ]
    assert sum(samples for _, samples in windows) == 2703
    assert all(offset % 16 == 0 for offset, _ in windows)
    assert all(samples % 16 == 0 for _, samples in windows[:-1])


def test_windows_limit_shards_to_original_chunks() -> None:
    assert _windows(15, 16, 8) == [(0, 15)]

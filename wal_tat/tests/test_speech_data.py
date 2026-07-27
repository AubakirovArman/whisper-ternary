import io

import numpy as np
import pytest
import soundfile as sf

from wal_tat import decode_audio_payload


def test_decode_audio_payload_uses_raw_bytes_and_mixes_stereo():
    buffer = io.BytesIO()
    stereo = np.stack((np.ones(160, dtype=np.float32), -np.ones(160, dtype=np.float32)), axis=1)
    sf.write(buffer, stereo, 16_000, format="WAV", subtype="FLOAT")
    audio, sampling_rate = decode_audio_payload({"bytes": buffer.getvalue(), "path": None})
    assert sampling_rate == 16_000
    assert audio.shape == (160,)
    assert np.max(np.abs(audio)) < 1e-6

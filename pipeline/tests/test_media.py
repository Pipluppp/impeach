from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from pipeline.media import (
    AcquisitionError,
    LocalFileMediaSource,
    YtDlpMediaSource,
    probe_audio,
    validate_requested_duration,
)


def test_probe_and_local_file_adapter(tmp_path: Path) -> None:
    audio = tmp_path / "tone.m4a"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=1",
            "-c:a",
            "aac",
            str(audio),
        ],
        check=True,
        timeout=30,
    )
    probe = probe_audio(audio)
    assert 0.9 <= probe.duration_seconds <= 1.1
    assert probe.codec_name == "aac"
    artifact = LocalFileMediaSource(audio).acquire(output_dir=tmp_path)
    assert artifact.adapter == "local-file"
    assert artifact.reused is True


def test_requested_duration_validation_rejects_bad_range(tmp_path: Path) -> None:
    with pytest.raises(AcquisitionError):
        YtDlpMediaSource("video").key(10, None)
    with pytest.raises(AcquisitionError):
        validate_requested_duration(
            type("Probe", (), {"duration_seconds": 5.0})(), 0, 100
        )

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

import pipeline.media as media_module

from pipeline.media import (
    AcquisitionError,
    AudioProbe,
    LocalFileMediaSource,
    YT_DLP_STRATEGIES,
    YtDlpMediaSource,
    probe_audio,
    validate_requested_duration,
)


def test_youtube_strategies_keep_credential_free_fallback_order() -> None:
    assert [strategy.name for strategy in YT_DLP_STRATEGIES] == [
        "default",
        "android_vr",
        "mweb_pot",
    ]
    assert YT_DLP_STRATEGIES[0].extractor_args is None
    assert YT_DLP_STRATEGIES[1].extractor_args == "youtube:player_client=android_vr"
    assert YT_DLP_STRATEGIES[2].extractor_args == "youtube:player_client=mweb"
    assert YT_DLP_STRATEGIES[2].extra_args == ("--js-runtimes", "node")


def test_youtube_acquisition_retries_with_android_vr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        if "youtube:player_client=android_vr" not in command:
            return subprocess.CompletedProcess(command, 1, "", "bot challenge")
        template = Path(command[command.index("-o") + 1])
        Path(str(template).replace("%(ext)s", "m4a")).write_bytes(b"test")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(media_module.subprocess, "run", fake_run)
    monkeypatch.setattr(
        media_module,
        "probe_audio",
        lambda _: AudioProbe(10.0, "aac", 44_100, 2),
    )

    artifact = YtDlpMediaSource("official-video").acquire(output_dir=tmp_path)

    assert artifact.adapter == "yt-dlp:android_vr"
    assert len(calls) == 2
    assert "youtube:player_client=android_vr" in calls[1]


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

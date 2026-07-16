"""Replaceable, temporary audio acquisition adapters with validation."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol


ROOT = Path(__file__).resolve().parents[1]


class AcquisitionError(RuntimeError):
    """Media acquisition failed without publishing a corrupt artifact."""


@dataclass(frozen=True)
class AudioProbe:
    duration_seconds: float
    codec_name: str
    sample_rate_hz: int | None
    channels: int | None


@dataclass(frozen=True)
class AudioArtifact:
    path: str
    adapter: str
    video_id: str | None
    requested_start: float | None
    requested_end: float | None
    probe: AudioProbe
    reused: bool
    acquisition_seconds: float


@dataclass(frozen=True)
class YtDlpStrategy:
    name: str
    extractor_args: str | None = None


# The default client is preferred. android_vr is the documented credential-free
# fallback that currently requires neither account cookies nor a YouTube PO token.
YT_DLP_STRATEGIES = (
    YtDlpStrategy("default"),
    YtDlpStrategy("android_vr", "youtube:player_client=android_vr"),
)


class MediaSource(Protocol):
    def acquire(
        self,
        *,
        output_dir: Path,
        start: float | None = None,
        end: float | None = None,
    ) -> AudioArtifact: ...


def probe_audio(path: Path) -> AudioProbe:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=30
        )
        payload = json.loads(completed.stdout)
        audio_stream = next(
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "audio"
        )
        duration = float(payload["format"]["duration"])
    except (FileNotFoundError, subprocess.SubprocessError, StopIteration, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise AcquisitionError(f"audio validation failed for {path.name}: {exc}") from exc
    if duration <= 0:
        raise AcquisitionError(f"audio validation reported non-positive duration for {path.name}")
    return AudioProbe(
        duration_seconds=duration,
        codec_name=audio_stream.get("codec_name", "unknown"),
        sample_rate_hz=int(audio_stream["sample_rate"]) if audio_stream.get("sample_rate") else None,
        channels=int(audio_stream["channels"]) if audio_stream.get("channels") else None,
    )


def relative_or_name(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT.resolve())).replace("\\", "/")
    except ValueError:
        return path.name


def validate_requested_duration(
    probe: AudioProbe, start: float | None, end: float | None
) -> None:
    if start is None or end is None:
        return
    requested = end - start
    if requested <= 0:
        raise AcquisitionError("section end must be greater than start")
    tolerance = max(5.0, requested * 0.02)
    if abs(probe.duration_seconds - requested) > tolerance:
        raise AcquisitionError(
            f"decoded duration {probe.duration_seconds:.2f}s differs from requested "
            f"{requested:.2f}s by more than {tolerance:.2f}s"
        )


class LocalFileMediaSource:
    def __init__(self, path: Path):
        self.path = path

    def acquire(
        self,
        *,
        output_dir: Path,
        start: float | None = None,
        end: float | None = None,
    ) -> AudioArtifact:
        if start is not None or end is not None:
            raise AcquisitionError("local-file adapter does not cut sections")
        probe = probe_audio(self.path)
        return AudioArtifact(
            path=relative_or_name(self.path),
            adapter="local-file",
            video_id=None,
            requested_start=None,
            requested_end=None,
            probe=probe,
            reused=True,
            acquisition_seconds=0.0,
        )


class YtDlpMediaSource:
    def __init__(self, video_id: str):
        self.video_id = video_id

    def key(self, start: float | None, end: float | None) -> str:
        if start is None and end is None:
            return f"{self.video_id}-full"
        if start is None or end is None:
            raise AcquisitionError("both section start and end are required")
        return f"{self.video_id}-{start:g}-{end:g}"

    def acquire(
        self,
        *,
        output_dir: Path,
        start: float | None = None,
        end: float | None = None,
    ) -> AudioArtifact:
        key = self.key(start, end)
        output_dir.mkdir(parents=True, exist_ok=True)
        final_path = output_dir / f"{key}.m4a"
        started = time.perf_counter()
        if final_path.is_file():
            probe = probe_audio(final_path)
            validate_requested_duration(probe, start, end)
            return AudioArtifact(
                path=relative_or_name(final_path),
                adapter="yt-dlp",
                video_id=self.video_id,
                requested_start=start,
                requested_end=end,
                probe=probe,
                reused=True,
                acquisition_seconds=round(time.perf_counter() - started, 3),
            )

        staging_root = output_dir / ".staging"
        staging_dir = staging_root / key
        if staging_dir.exists():
            resolved = staging_dir.resolve()
            if not resolved.is_relative_to(output_dir.resolve()):
                raise AcquisitionError("refusing to clean staging path outside output directory")
            shutil.rmtree(staging_dir)
        staging_dir.mkdir(parents=True)
        template = staging_dir / f"{key}.%(ext)s"
        command_base = [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--retries",
            "2",
            "--fragment-retries",
            "2",
            "--retry-sleep",
            "linear=1::2",
            "--sleep-requests",
            "0.25",
            "-f",
            "bestaudio/best",
            "--extract-audio",
            "--audio-format",
            "m4a",
            "-o",
            str(template),
        ]
        if start is not None and end is not None:
            command_base.extend(["--download-sections", f"*{start:g}-{end:g}"])

        try:
            failures: list[str] = []
            for strategy in YT_DLP_STRATEGIES:
                shutil.rmtree(staging_dir)
                staging_dir.mkdir(parents=True)
                command = list(command_base)
                if strategy.extractor_args:
                    command.extend(["--extractor-args", strategy.extractor_args])
                command.append(f"https://www.youtube.com/watch?v={self.video_id}")
                completed = subprocess.run(
                    command, capture_output=True, text=True, timeout=7200
                )
                if completed.returncode != 0:
                    detail = (completed.stderr or completed.stdout).strip().splitlines()
                    failures.append(
                        f"{strategy.name}: "
                        + (detail[-1] if detail else f"exit {completed.returncode}")
                    )
                    continue
                candidates = [
                    path
                    for path in staging_dir.iterdir()
                    if path.is_file() and path.suffix.casefold() == ".m4a"
                ]
                if len(candidates) != 1:
                    failures.append(
                        f"{strategy.name}: produced {len(candidates)} m4a candidates"
                    )
                    continue
                probe = probe_audio(candidates[0])
                validate_requested_duration(probe, start, end)
                candidates[0].replace(final_path)
                return AudioArtifact(
                    path=relative_or_name(final_path),
                    adapter=f"yt-dlp:{strategy.name}",
                    video_id=self.video_id,
                    requested_start=start,
                    requested_end=end,
                    probe=probe,
                    reused=False,
                    acquisition_seconds=round(time.perf_counter() - started, 3),
                )
            raise AcquisitionError(
                "yt-dlp failed for all credential-free clients: " + "; ".join(failures)
            )
        finally:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if staging_root.exists() and not any(staging_root.iterdir()):
                staging_root.rmdir()


def artifact_json(artifact: AudioArtifact) -> dict[str, Any]:
    payload = asdict(artifact)
    payload["state"] = "audio_acquired"
    return payload


def write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    acquire = subparsers.add_parser("acquire")
    acquire.add_argument("--video-id", required=True)
    acquire.add_argument("--output-dir", type=Path, default=ROOT / ".cache" / "media")
    acquire.add_argument("--start", type=float)
    acquire.add_argument("--end", type=float)
    acquire.add_argument("--status-file", type=Path)
    args = parser.parse_args(argv)

    source = YtDlpMediaSource(args.video_id)
    try:
        artifact = source.acquire(output_dir=args.output_dir, start=args.start, end=args.end)
        payload = artifact_json(artifact)
        if args.status_file:
            write_status(args.status_file, payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except AcquisitionError as exc:
        failure = {
            "state": "audio_acquisition_failed",
            "adapter": "yt-dlp",
            "video_id": args.video_id,
            "error": str(exc),
            "resumable": True,
        }
        if args.status_file:
            write_status(args.status_file, failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

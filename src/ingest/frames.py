"""Frame sampling — ffmpeg piped straight to memory, no write-then-reopen.

One ffmpeg pass decodes, samples (interval or scene-cut), downscales to
THUMB_WIDTH and emits an MJPEG stream on stdout; we split it into individual
JPEGs by SOI/EOI markers. Nothing frame-shaped ever touches local disk — the
JPEG bytes go on to dedup, CLIP, and object storage as-is.

Sampling is the single biggest scaling lever: it is what stops thousands of
videos from becoming billions of near-identical vectors.
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .. import config  # read at call-time so seeding can use lighter sampling

_JPEG_SOI = b"\xff\xd8"
_JPEG_EOI = b"\xff\xd9"
_PTS_RE = re.compile(r"pts_time:([0-9.]+)")


@dataclass
class Frame:
    ms: int        # position in the video
    jpeg: bytes    # downscaled JPEG, ready for CLIP + thumbnail upload


def _split_mjpeg(stream: bytes) -> list[bytes]:
    """Split a concatenated MJPEG byte stream into individual JPEG images."""
    frames: list[bytes] = []
    pos = 0
    while True:
        start = stream.find(_JPEG_SOI, pos)
        if start < 0:
            break
        end = stream.find(_JPEG_EOI, start + 2)
        if end < 0:
            break
        frames.append(stream[start:end + 2])
        pos = end + 2
    return frames


def probe_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, text=True,
    )
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0


def _run_ffmpeg(path: Path, vf: str) -> tuple[bytes, str]:
    """One decode pass -> (mjpeg stdout, stderr text for showinfo parsing)."""
    proc = subprocess.run(
        ["ffmpeg", "-loglevel", "info", "-i", str(path),
         "-vf", vf, "-vsync", "vfr",
         "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", str(config.THUMB_QUALITY), "-"],
        capture_output=True,
    )
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='replace')[-500:]}")
    return proc.stdout, proc.stderr.decode(errors="replace")


def _scale() -> str:
    # Downscale in the same decode pass; -2 keeps the height even for mjpeg.
    return f"scale='min({config.THUMB_WIDTH},iw)':-2"


def _sample_interval(path: Path) -> list[Frame]:
    """One frame every FRAME_INTERVAL_SEC (spacing widened to respect MAX_FRAMES)."""
    interval = max(0.2, config.FRAME_INTERVAL_SEC)
    duration = probe_duration(path)
    if config.MAX_FRAMES and duration > 0 and duration / interval > config.MAX_FRAMES:
        interval = duration / config.MAX_FRAMES
    stdout, _ = _run_ffmpeg(path, f"fps=1/{interval:.6f},{_scale()}")
    jpegs = _split_mjpeg(stdout)
    return [Frame(ms=int(round(i * interval * 1000)), jpeg=j)
            for i, j in enumerate(jpegs)]


def _sample_scene(path: Path) -> list[Frame]:
    """One frame per detected scene cut; timestamps parsed from showinfo."""
    stdout, stderr = _run_ffmpeg(
        path, f"select='gt(scene,{config.SCENE_THRESHOLD})',showinfo,{_scale()}")
    jpegs = _split_mjpeg(stdout)
    times = [int(round(float(m) * 1000)) for m in _PTS_RE.findall(stderr)]
    if len(times) >= len(jpegs):
        frames = [Frame(ms=t, jpeg=j) for t, j in zip(times, jpegs)]
    else:  # showinfo parsing fell short — degrade to index-based stamps
        frames = [Frame(ms=i * 1000, jpeg=j) for i, j in enumerate(jpegs)]
    if config.MAX_FRAMES and len(frames) > config.MAX_FRAMES:
        step = len(frames) / config.MAX_FRAMES
        frames = [frames[int(i * step)] for i in range(config.MAX_FRAMES)]
    return frames


def sample(path: Path) -> list[Frame]:
    if config.FRAME_STRATEGY == "scene":
        return _sample_scene(path)
    return _sample_interval(path)

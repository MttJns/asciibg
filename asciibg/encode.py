"""Write rendered frames out as video, GIF and a still."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def require_ffmpeg() -> str:
    """Return the ffmpeg binary path, or raise if it is not installed."""
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError(
            "ffmpeg not found on PATH. Frames are piped straight into it, so it has "
            "to be installed:\n"
            "  macOS          brew install ffmpeg\n"
            "  Debian/Ubuntu  sudo apt-get install ffmpeg\n"
            "  Windows        winget install Gyan.FFmpeg"
        )
    return path


class Mp4Writer:
    """Pipe raw RGB frames into a Zoom-compatible H.264 file."""

    def __init__(self, path: Path, width: int, height: int, fps: int, crf: int = 18) -> None:
        self.path = path
        cmd = [
            require_ffmpeg(), "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
            "-an",
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(path),
        ]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.write(frame.tobytes())

    def close(self) -> None:
        assert self._proc.stdin is not None
        self._proc.stdin.close()
        if self._proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")

    def __enter__(self) -> "Mp4Writer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def mp4_to_gif(src: Path, dst: Path, *, width: int = 720, fps: int = 15) -> None:
    """Make a looping GIF from the rendered video, with its own 256-colour palette."""
    vf = (
        f"[0:v] fps={fps},scale={width}:-1:flags=lanczos,split [a][b];"
        " [a] palettegen=stats_mode=full [p];"
        " [b][p] paletteuse=dither=bayer:bayer_scale=3"
    )
    cmd = [require_ffmpeg(), "-y", "-loglevel", "error", "-i", str(src),
           "-filter_complex", vf, "-loop", "0", str(dst)]
    subprocess.run(cmd, check=True)


def save_png(frame: np.ndarray, path: Path) -> None:
    """Save a single frame as a still background."""
    Image.fromarray(frame).save(path)


def probe(path: Path) -> Optional[str]:
    """Return a one-line summary of a written video, or None if ffprobe is missing."""
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    out = subprocess.run(
        [exe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name,width,height,pix_fmt,r_frame_rate,nb_frames",
         "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    return " ".join(out)

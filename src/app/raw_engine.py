import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RawEngine(str, Enum):
    AUTO = "Auto (Recommended)"
    CAMERA_PREVIEW = "Camera Preview"
    RAWPY = "LibRaw/rawpy"
    RAWTHERAPEE = "RawTherapee CLI"
    DARKTABLE = "darktable-cli"


@dataclass(frozen=True)
class DetectedRawEngines:
    rawtherapee: Optional[str]
    darktable: Optional[str]


def _first_existing_path(candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def detect_raw_engines(
    rawtherapee_path: Optional[str] = None,
    darktable_path: Optional[str] = None,
) -> DetectedRawEngines:
    """Detect optional external RAW engines from explicit paths, PATH, and common Windows installs."""
    rawtherapee_candidates = [
        rawtherapee_path or "",
        shutil.which("rawtherapee-cli") or "",
        shutil.which("rawtherapee-cli.exe") or "",
        r"C:\Program Files\RawTherapee\rawtherapee-cli.exe",
        r"C:\Program Files\RawTherapee-5.11\rawtherapee-cli.exe",
        r"C:\Program Files\RawTherapee-5.10\rawtherapee-cli.exe",
        r"C:\Program Files\RawTherapee-5.9\rawtherapee-cli.exe",
    ]
    darktable_candidates = [
        darktable_path or "",
        shutil.which("darktable-cli") or "",
        shutil.which("darktable-cli.exe") or "",
        r"C:\Program Files\darktable\bin\darktable-cli.exe",
    ]
    return DetectedRawEngines(
        rawtherapee=_first_existing_path(rawtherapee_candidates),
        darktable=_first_existing_path(darktable_candidates),
    )


def parse_raw_engine(value: str) -> RawEngine:
    for engine in RawEngine:
        if value == engine.value or value == engine.name:
            return engine
    return RawEngine.AUTO


def select_raw_engine(
    requested: RawEngine,
    *,
    has_safe_preview: bool,
    rawtherapee_path: Optional[str],
    darktable_path: Optional[str],
) -> RawEngine:
    """Choose the actual engine to use for a RAW file."""
    if requested != RawEngine.AUTO:
        if requested == RawEngine.CAMERA_PREVIEW and not has_safe_preview:
            return RawEngine.RAWPY
        if requested == RawEngine.RAWTHERAPEE and not rawtherapee_path:
            return RawEngine.RAWPY
        if requested == RawEngine.DARKTABLE and not darktable_path:
            return RawEngine.RAWPY
        return requested

    if has_safe_preview:
        return RawEngine.CAMERA_PREVIEW
    if rawtherapee_path:
        return RawEngine.RAWTHERAPEE
    if darktable_path:
        return RawEngine.DARKTABLE
    return RawEngine.RAWPY


def build_rawtherapee_command(executable: str, input_path: str, output_path: str, quality: int = 92) -> list[str]:
    """Build RawTherapee CLI command for direct JPEG export."""
    return [
        executable,
        "-o",
        output_path,
        f"-j{quality}",
        "-Y",
        "-c",
        input_path,
    ]


def build_darktable_command(executable: str, input_path: str, output_path: str, quality: int = 92) -> list[str]:
    """Build darktable-cli command for direct JPEG export."""
    return [
        executable,
        input_path,
        output_path,
        "--core",
        "--conf",
        f"plugins/imageio/format/jpeg/quality={quality}",
    ]



def run_external_raw_engine(command: list[str]) -> tuple[bool, str]:
    startupinfo = None
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = subprocess.SW_HIDE
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            startupinfo=startupinfo,
            check=False,
        )
        if result.returncode == 0:
            return True, "External RAW engine completed successfully."
        stderr = (result.stderr or result.stdout or "").strip()
        return False, f"External RAW engine failed with exit code {result.returncode}: {stderr}"
    except Exception as exc:
        return False, f"External RAW engine error: {exc}"

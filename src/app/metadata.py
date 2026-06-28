import subprocess
import shutil
import os
from typing import Tuple

def is_exiftool_available() -> bool:
    """Checks if exiftool is available in the system PATH."""
    return shutil.which("exiftool") is not None

def copy_exif_metadata(original_path: str, target_path: str) -> Tuple[bool, str]:
    """
    Copies all EXIF metadata from original_path to target_path using exiftool if available.
    Returns:
        Tuple[bool, str]: (success, status_message)
    """
    if not is_exiftool_available():
        return False, "ExifTool not found; basic metadata only."

    if not os.path.exists(original_path):
        return False, f"Original file not found: {original_path}"
    if not os.path.exists(target_path):
        return False, f"Target file not found: {target_path}"

    cmd = [
        "exiftool",
        "-overwrite_original",
        "-TagsFromFile",
        original_path,
        "-all:all",
        target_path
    ]

    try:
        # Run command hidden / with no window on Windows if subprocess is used
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            startupinfo=startupinfo,
            check=False
        )

        if result.returncode == 0:
            return True, "Metadata copied successfully using ExifTool."
        else:
            err_msg = result.stderr.strip() if result.stderr else f"Exit code {result.returncode}"
            return False, f"ExifTool failed: {err_msg}"
            
    except Exception as e:
        return False, f"Error running ExifTool: {str(e)}"

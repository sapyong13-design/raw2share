import os
import re
import subprocess
import time
from typing import Tuple, List, Callable, Optional
import imageio_ffmpeg

def get_ffmpeg_path() -> str:
    """Retrieve ffmpeg path from imageio_ffmpeg or system fallback."""
    try:
        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.exists(path):
            return path
    except Exception:
        pass
    return "ffmpeg"

def build_ffmpeg_command(
    input_path: str,
    output_path: str,
    preset_name: str,
    fps: str,
    faststart: bool = True
) -> List[str]:
    """
    Builds the FFmpeg command list for the chosen parameters.
    """
    ffmpeg_exe = get_ffmpeg_path()
    cmd = [ffmpeg_exe, "-y", "-i", input_path]

    # Video filters (scaling)
    vf_filters = []
    
    # Presets mapping
    # Default is 'WhatsApp Balanced 1080p'
    crf = 23
    preset = "medium"
    audio_bitrate = "160k"
    
    if preset_name == "High Quality 1080p":
        crf = 20
        preset = "slow"
        audio_bitrate = "192k"
        vf_filters.append("scale='if(gt(iw,1920),1920,iw)':-2")
    elif preset_name == "Small File 720p":
        crf = 26
        preset = "medium"
        audio_bitrate = "128k"
        vf_filters.append("scale='if(gt(iw,1280),1280,iw)':-2")
    elif preset_name == "Original Resolution High Quality":
        crf = 20
        preset = "slow"
        audio_bitrate = "192k"
        # No scaling
    else: # WhatsApp Balanced 1080p
        crf = 23
        preset = "medium"
        audio_bitrate = "160k"
        vf_filters.append("scale='if(gt(iw,1920),1920,iw)':-2")

    # Add frame rate if requested
    if fps == "30 fps":
        cmd.extend(["-r", "30"])
    elif fps == "60 fps":
        cmd.extend(["-r", "60"])

    # Add video filter arguments if any
    if vf_filters:
        cmd.extend(["-vf", ",".join(vf_filters)])

    # Video codec settings
    cmd.extend([
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p"  # Ensure compatibility with WhatsApp/Android/iOS
    ])

    # Audio settings
    cmd.extend([
        "-c:a", "aac",
        "-b:a", audio_bitrate
    ])

    # Faststart
    if faststart:
        cmd.extend(["-movflags", "+faststart"])

    # Map metadata from the original file
    cmd.extend(["-map_metadata", "0"])

    # Output path
    cmd.append(output_path)
    return cmd

def parse_time_to_seconds(time_str: str) -> float:
    """Parses HH:MM:SS.xx to total seconds."""
    try:
        parts = time_str.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return float(h) * 3600 + float(m) * 60 + float(s)
    except Exception:
        pass
    return 0.0

def convert_video(
    input_path: str,
    output_path: str,
    settings: dict,
    progress_callback: Optional[Callable[[float], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None
) -> dict:
    """
    Executes FFmpeg conversion and reports progress.
    
    settings dict keys:
    - preset: str ('WhatsApp Balanced 1080p', etc.)
    - fps: str ('Keep original', '30 fps', '60 fps')
    - faststart: bool (default True)
    
    Returns:
        dict: {'success': bool, 'message': str, 'size_before': int, 'size_after': int}
    """
    try:
        if not os.path.exists(input_path):
            return {
                'success': False,
                'message': f"Input file not found: {input_path}",
                'size_before': 0,
                'size_after': 0
            }

        size_before = os.path.getsize(input_path)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        preset = settings.get('preset', 'WhatsApp Balanced 1080p')
        fps = settings.get('fps', 'Keep original')
        faststart = settings.get('faststart', True)

        cmd = build_ffmpeg_command(input_path, output_path, preset, fps, faststart)

        # Setup startup info to hide terminal window on Windows
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE

        # We start FFmpeg. FFmpeg output is sent to stderr by default.
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # merge stderr into stdout to read from a single pipe
            stdin=subprocess.PIPE,     # allow sending 'q' to stop or pipe input
            text=True,
            bufsize=1,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        )

        total_duration = 0.0
        current_time = 0.0

        # Regular expressions for duration and time progress
        duration_regex = re.compile(r"Duration:\s*(\d{2}:\d{2}:\d{2}\.\d{2})")
        time_regex = re.compile(r"time=\s*(\d{2}:\d{2}:\d{2}\.\d{2})")

        # Read output line by line
        while True:
            # Check for cancellation
            if cancel_check and cancel_check():
                # Terminate FFmpeg gracefully by writing 'q'
                try:
                    process.stdin.write('q')
                    process.stdin.flush()
                except Exception:
                    pass
                # Wait up to 2 seconds then terminate if still running
                time.sleep(0.5)
                if process.poll() is None:
                    process.terminate()
                return {
                    'success': False,
                    'message': "Cancelled by user.",
                    'size_before': size_before,
                    'size_after': 0
                }

            line = process.stdout.readline()
            if not line:
                break

            # Parse duration from the output
            if total_duration == 0.0:
                duration_match = duration_regex.search(line)
                if duration_match:
                    total_duration = parse_time_to_seconds(duration_match.group(1))

            # Parse current time from the progress updates
            time_match = time_regex.search(line)
            if time_match:
                current_time = parse_time_to_seconds(time_match.group(1))
                if total_duration > 0.0 and progress_callback:
                    progress_pct = min(100.0, (current_time / total_duration) * 100.0)
                    progress_callback(progress_pct)

        process.wait()

        if process.returncode == 0 and os.path.exists(output_path):
            size_after = os.path.getsize(output_path)
            return {
                'success': True,
                'message': "Video converted successfully.",
                'size_before': size_before,
                'size_after': size_after
            }
        else:
            return {
                'success': False,
                'message': f"FFmpeg failed with exit code {process.returncode}.",
                'size_before': size_before,
                'size_after': 0
            }

    except Exception as e:
        return {
            'success': False,
            'message': f"Error during video conversion: {str(e)}",
            'size_before': os.path.getsize(input_path) if os.path.exists(input_path) else 0,
            'size_after': 0
        }

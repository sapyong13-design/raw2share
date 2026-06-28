import os
from typing import List, Tuple, Set

PHOTO_EXTENSIONS = {'.cr3', '.cr2', '.jpg', '.jpeg'}
VIDEO_EXTENSIONS = {'.mov', '.mp4', '.m4v', '.mts'}

def is_photo(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in PHOTO_EXTENSIONS

def is_video(filename: str) -> bool:
    _, ext = os.path.splitext(filename)
    return ext.lower() in VIDEO_EXTENSIONS

def scan_path(
    path: str,
    recursive: bool,
    output_dir: str = None
) -> Tuple[List[str], List[str]]:
    """
    Scans a file or folder for raw photos and videos.
    Returns:
        Tuple[List[str], List[str]]: (photos, videos)
    """
    photos: List[str] = []
    videos: List[str] = []
    seen: Set[str] = set()

    # Normalize paths for comparison
    output_dir_norm = os.path.abspath(output_dir) if output_dir else None

    def add_file(f_path: str):
        abs_path = os.path.abspath(f_path)
        if abs_path in seen:
            return
        
        # Avoid scanning files that are inside the output directory
        if output_dir_norm and abs_path.startswith(output_dir_norm):
            return

        seen.add(abs_path)
        if is_photo(f_path):
            photos.append(abs_path)
        elif is_video(f_path):
            videos.append(abs_path)

    if os.path.isfile(path):
        add_file(path)
    elif os.path.isdir(path):
        if recursive:
            for root, _, files in os.walk(path):
                # Skip output directory subtree if output_dir is a subdirectory of input
                if output_dir_norm and os.path.abspath(root).startswith(output_dir_norm):
                    continue
                for file in files:
                    add_file(os.path.join(root, file))
        else:
            try:
                for entry in os.scandir(path):
                    if entry.is_file():
                        add_file(entry.path)
            except OSError:
                pass

    return photos, videos

def scan_paths(
    paths: List[str],
    recursive: bool,
    output_dir: str = None
) -> Tuple[List[str], List[str]]:
    """Scan multiple paths (files/folders) and return unified lists of photos and videos."""
    all_photos: List[str] = []
    all_videos: List[str] = []
    seen: Set[str] = set()

    for p in paths:
        p_photos, p_videos = scan_path(p, recursive, output_dir)
        for ph in p_photos:
            if ph not in seen:
                seen.add(ph)
                all_photos.append(ph)
        for vd in p_videos:
            if vd not in seen:
                seen.add(vd)
                all_videos.append(vd)

    return all_photos, all_videos


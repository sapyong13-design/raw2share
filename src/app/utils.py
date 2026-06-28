import os

def format_size(size_bytes: int) -> str:
    """Format file size in bytes to a human-readable string."""
    if size_bytes < 0:
        return "N/A"
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"

def get_unique_filepath(target_path: str) -> str:
    """Generate a unique filename by appending _1, _2, etc. if conflict exists."""
    if not os.path.exists(target_path):
        return target_path
    
    dir_name, file_name = os.path.split(target_path)
    base_name, ext = os.path.splitext(file_name)
    
    counter = 1
    while True:
        new_file_name = f"{base_name}_{counter}{ext}"
        new_path = os.path.join(dir_name, new_file_name)
        if not os.path.exists(new_path):
            return new_path
        counter += 1

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


def is_legacy_pc() -> bool:
    """Detect if the current PC is considered legacy/low-spec (e.g. <= 4 CPU cores or <= 8GB RAM)."""
    import os
    import sys
    # 1. Check CPU cores
    cpu_count = os.cpu_count() or 4
    if cpu_count <= 4:
        return True
        
    # 2. Check RAM on Windows
    if sys.platform == "win32":
        try:
            import ctypes
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            total_ram_gb = stat.ullTotalPhys / (1024.0 ** 3)
            if total_ram_gb <= 8.5:  # Allow small buffer for 8GB RAM
                return True
        except Exception:
            pass
            
    return False

import sys
from PySide6.QtWidgets import QApplication
from app.gui import MainWindow

def main():
    try:
        import cv2
        import os
        from app.utils import is_legacy_pc
        # Enable GPU OpenCL acceleration in OpenCV
        cv2.ocl.setUseOpenCL(True)
        # Limit CPU thread overhead on legacy machines to prevent thrashing
        if is_legacy_pc():
            cores = os.cpu_count() or 4
            cv2.setNumThreads(max(1, cores - 1))
    except Exception:
        pass
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

import os
from PySide6.QtCore import QThread, Signal
from app.photo_converter import convert_raw_to_jpg
from app.video_converter import convert_video
from app.utils import get_unique_filepath, format_size
from app.file_scanner import is_photo, is_video

class ConversionWorker(QThread):
    # Signals to communicate with the GUI thread
    file_started = Signal(int, str)  # (index, input_path)
    file_progress = Signal(int, float)  # (index, progress_percent)
    file_completed = Signal(int, bool, str, int, int, str)  # (index, success, message, size_before, size_after, output_path)
    all_completed = Signal()
    log_message = Signal(str)

    def __init__(self, queue: list, settings: dict):
        """
        queue: List of dicts representing each file in the queue:
               {
                   'index': int,
                   'input_path': str,
                   'base_dir': str,
                   'type': str ('Photo' or 'Video')
               }
        settings: Dict of configuration settings from the GUI
        """
        super().__init__()
        self.queue = queue
        self.settings = settings
        self._is_cancelled = False
        self.current_process_cancel = False

    def cancel(self):
        self._is_cancelled = True
        self.current_process_cancel = True

    def _parse_max_dimension(self, value):
        try:
            if not value or str(value).lower() == 'original':
                return 0
            return int(value)
        except (TypeError, ValueError):
            return 0

    def check_cancel(self) -> bool:
        return self._is_cancelled

    def run(self):
        self.log_message.emit("Batch processing started.")
        
        for item in self.queue:
            if self._is_cancelled:
                self.log_message.emit("Batch processing cancelled by user.")
                break

            idx = item['index']
            input_path = item['input_path']
            base_dir = item['base_dir']
            file_type = item['type']

            self.file_started.emit(idx, input_path)

            # Determine the target output folder
            user_output_dir = self.settings.get('output_dir', '')
            preserve_structure = self.settings.get('preserve_structure', False)
            create_subfolders = self.settings.get('create_subfolders', False)

            # 1. Fallback to default output folder if not specified
            if not user_output_dir:
                # default output folder: input folder + "/RAW2Share_Output"
                target_base = os.path.join(os.path.dirname(input_path), "RAW2Share_Output")
            else:
                target_base = user_output_dir

            # 2. Preserve folder structure relative to scanned directory
            rel_dir = ""
            if preserve_structure and base_dir:
                try:
                    rel_dir = os.path.relpath(os.path.dirname(input_path), base_dir)
                    if rel_dir == ".":
                        rel_dir = ""
                except Exception:
                    pass

            # 3. Create subfolders based on file type
            subfolder = ""
            if create_subfolders:
                if file_type == 'Photo':
                    subfolder = "Photos_JPG"
                else:
                    subfolder = "Videos_WhatsApp"

            final_dir = os.path.abspath(os.path.join(target_base, rel_dir, subfolder))

            # 4. Generate base output file name with suffix and target extension
            base_name, _ = os.path.splitext(os.path.basename(input_path))
            
            if file_type == 'Photo':
                suffix = self.settings.get('photo_suffix', '')
                out_ext = ".jpg"
            else:
                suffix = self.settings.get('video_suffix', '_WA1080p')
                out_ext = ".mp4"

            dest_filename = f"{base_name}{suffix}{out_ext}"
            target_path = os.path.join(final_dir, dest_filename)

            # 5. Resolve file name conflict
            target_path = get_unique_filepath(target_path)

            # 6. Execute conversion
            result = None
            if file_type == 'Photo':
                self.log_message.emit(f"Converting photo: {os.path.basename(input_path)} -> {os.path.basename(target_path)}")
                photo_settings = {
                    'quality': self.settings.get('photo_quality', 98),
                    'keep_resolution': self.settings.get('photo_keep_resolution', True),
                    'autocorrect_mode': self.settings.get('photo_autocorrect', 'Smart Auto (Recommended)'),
                    'use_camera_wb': self.settings.get('photo_camera_wb', True),
                    'copy_exif': self.settings.get('photo_copy_exif', True),
                    'correct_corner_shading': self.settings.get('photo_correct_corner_shading', True),
                    'raw_engine': self.settings.get('raw_engine', 'Auto (Recommended)'),
                    'max_dimension': self._parse_max_dimension(self.settings.get('photo_max_dimension', 'Original')),
                    'manual_adjustments': self.settings.get('manual_adjustments', {})
                }
                
                # Photos conversion is typically fast, so we check cancel just before/after
                if self._is_cancelled:
                    break
                
                result = convert_raw_to_jpg(input_path, target_path, photo_settings)
                self.file_progress.emit(idx, 100.0)
            else:
                self.log_message.emit(f"Converting video: {os.path.basename(input_path)} -> {os.path.basename(target_path)}")
                video_settings = {
                    'preset': self.settings.get('video_preset', 'WhatsApp Balanced 1080p'),
                    'fps': self.settings.get('video_fps', 'Keep original'),
                    'faststart': self.settings.get('video_faststart', True)
                }

                def prog_cb(pct):
                    self.file_progress.emit(idx, pct)

                result = convert_video(
                    input_path,
                    target_path,
                    video_settings,
                    progress_callback=prog_cb,
                    cancel_check=self.check_cancel
                )

            # Handle the result
            if result:
                success = result.get('success', False)
                msg = result.get('message', '')
                size_before = result.get('size_before', 0)
                size_after = result.get('size_after', 0)
                
                self.file_completed.emit(idx, success, msg, size_before, size_after, target_path if success else "")
                self.log_message.emit(f"Finished {os.path.basename(input_path)}: {msg}")
            else:
                self.file_completed.emit(idx, False, "Unknown failure", 0, 0, "")
                self.log_message.emit(f"Finished {os.path.basename(input_path)}: Failed (Unknown error)")

        self.all_completed.emit()
        self.log_message.emit("Batch processing finished.")


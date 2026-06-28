import os
import sys
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QGroupBox, QPushButton, QCheckBox, QSlider, QComboBox, QLineEdit,
    QLabel, QTableWidget, QTableWidgetItem, QProgressBar, QFileDialog,
    QMessageBox, QTextEdit, QSplitter, QHeaderView, QAbstractItemView
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from app.file_scanner import scan_paths, is_photo, is_video
from app.workers import ConversionWorker
from app.utils import format_size
from app.raw_engine import RawEngine, detect_raw_engines

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RAW2Share")
        self.setMinimumSize(1024, 700)
        
        # Enable Drag and Drop on the main window
        self.setAcceptDrops(True)
        
        self.scanned_files = [] # list of dicts: {'input_path': str, 'base_dir': str, 'type': str}
        self.processing_worker = None
        self.selected_output_dir = ""
        self.logs = []

        self.init_ui()

    def log(self, message: str):
        self.logs.append(message)
        self.log_display.append(message)

    def init_ui(self):
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Splitter to separate controls and table/logs
        main_splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(main_splitter)

        # Top section: Input, Output, and Settings
        top_widget = QWidget()
        top_layout = QHBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        main_splitter.addWidget(top_widget)

        # Top-Left: Input/Output Paths
        path_group = QGroupBox("Input / Output")
        path_grid = QGridLayout(path_group)
        
        # Input row
        self.btn_select_files = QPushButton("Select Files")
        self.btn_select_files.clicked.connect(self.select_files)
        self.btn_select_folder = QPushButton("Select Folder")
        self.btn_select_folder.clicked.connect(self.select_folder)
        
        self.cb_recursive = QCheckBox("Scan Folders Recursively")
        self.cb_recursive.setChecked(True)

        path_grid.addWidget(self.btn_select_files, 0, 0)
        path_grid.addWidget(self.btn_select_folder, 0, 1)
        path_grid.addWidget(self.cb_recursive, 0, 2)

        # Output row
        self.btn_select_output = QPushButton("Select Output Folder")
        self.btn_select_output.clicked.connect(self.select_output_folder)
        
        self.lbl_output_path = QLabel("Default: input folder + '/RAW2Share_Output'")
        self.lbl_output_path.setWordWrap(True)
        self.lbl_output_path.setStyleSheet("color: #666; font-style: italic;")

        self.cb_preserve_structure = QCheckBox("Preserve Folder Structure")
        self.cb_preserve_structure.setChecked(False)
        
        self.cb_create_subfolders = QCheckBox("Create Subfolders (Photos_JPG / Videos_WhatsApp)")
        self.cb_create_subfolders.setChecked(False)

        path_grid.addWidget(self.btn_select_output, 1, 0)
        path_grid.addWidget(self.lbl_output_path, 1, 1, 1, 2)
        path_grid.addWidget(self.cb_preserve_structure, 2, 0, 1, 3)
        path_grid.addWidget(self.cb_create_subfolders, 3, 0, 1, 3)

        top_layout.addWidget(path_group, 3)

        # Top-Middle: Photo Settings
        photo_group = QGroupBox("Photo Settings")
        photo_layout = QGridLayout(photo_group)

        self.cb_convert_photos = QCheckBox("Convert RAW photos to JPG")
        self.cb_convert_photos.setChecked(True)
        photo_layout.addWidget(self.cb_convert_photos, 0, 0, 1, 2)

        # Quality Slider
        photo_layout.addWidget(QLabel("JPG Quality:"), 1, 0)
        self.slider_quality = QSlider(Qt.Horizontal)
        self.slider_quality.setRange(90, 100)
        self.slider_quality.setValue(92)
        self.lbl_quality_val = QLabel("92")
        self.slider_quality.valueChanged.connect(lambda val: self.lbl_quality_val.setText(str(val)))
        
        quality_hb = QHBoxLayout()
        quality_hb.addWidget(self.slider_quality)
        quality_hb.addWidget(self.lbl_quality_val)
        photo_layout.addLayout(quality_hb, 1, 1)

        photo_layout.addWidget(QLabel("Export preset:"), 2, 0)
        self.cmb_photo_export_preset = QComboBox()
        self.cmb_photo_export_preset.addItems([
            "Archive High Quality",
            "Office Share Balanced",
            "WhatsApp / Social",
            "Custom",
        ])
        self.cmb_photo_export_preset.setCurrentText("Office Share Balanced")
        self.cmb_photo_export_preset.currentTextChanged.connect(self.apply_photo_export_preset)
        photo_layout.addWidget(self.cmb_photo_export_preset, 2, 1)

        photo_layout.addWidget(QLabel("Max dimension:"), 3, 0)
        self.cmb_photo_max_dimension = QComboBox()
        self.cmb_photo_max_dimension.addItems(["Original", "4096", "3000", "2560", "1920"])
        self.cmb_photo_max_dimension.setCurrentText("3000")
        photo_layout.addWidget(self.cmb_photo_max_dimension, 3, 1)

        self.cb_auto_each_photo = QCheckBox("Auto-correct each photo individually")
        self.cb_auto_each_photo.setChecked(True)
        photo_layout.addWidget(self.cb_auto_each_photo, 4, 0, 1, 2)

        # Keep Full Resolution Checkbox
        self.cb_photo_resolution = QCheckBox("Keep original resolution")
        self.cb_photo_resolution.setChecked(False)
        photo_layout.addWidget(self.cb_photo_resolution, 5, 0, 1, 2)

        # Autocorrect Mode
        photo_layout.addWidget(QLabel("Autocorrect:"), 6, 0)
        self.cmb_autocorrect = QComboBox()
        self.cmb_autocorrect.addItems(["Smart Auto (Recommended)", "Off", "Natural", "Bright", "Vivid", "Low Light"])
        photo_layout.addWidget(self.cmb_autocorrect, 6, 1)

        # Camera WB & Copy EXIF
        self.cb_camera_wb = QCheckBox("Use Camera White Balance")
        self.cb_camera_wb.setChecked(True)
        photo_layout.addWidget(self.cb_camera_wb, 7, 0, 1, 2)

        self.cb_copy_exif = QCheckBox("Copy EXIF Metadata")
        self.cb_copy_exif.setChecked(True)
        photo_layout.addWidget(self.cb_copy_exif, 8, 0, 1, 2)

        self.cb_raw_corner_shading = QCheckBox("Correct RAW corner shading / vignette")
        self.cb_raw_corner_shading.setChecked(True)
        photo_layout.addWidget(self.cb_raw_corner_shading, 9, 0, 1, 2)

        photo_layout.addWidget(QLabel("RAW Engine:"), 10, 0)
        self.cmb_raw_engine = QComboBox()
        self.cmb_raw_engine.addItems([engine.value for engine in RawEngine])
        photo_layout.addWidget(self.cmb_raw_engine, 10, 1)

        detected_engines = detect_raw_engines()
        engine_status = []
        engine_status.append("RawTherapee: Found" if detected_engines.rawtherapee else "RawTherapee: Not found")
        engine_status.append("darktable: Found" if detected_engines.darktable else "darktable: Not found")
        self.lbl_raw_engine_status = QLabel(" | ".join(engine_status))
        self.lbl_raw_engine_status.setStyleSheet("color: #666; font-size: 11px;")
        self.lbl_raw_engine_status.setWordWrap(True)
        photo_layout.addWidget(self.lbl_raw_engine_status, 11, 0, 1, 2)

        # Photo suffix
        photo_layout.addWidget(QLabel("Photo suffix:"), 12, 0)
        self.txt_photo_suffix = QLineEdit("")
        photo_layout.addWidget(self.txt_photo_suffix, 12, 1)

        manual_group = QGroupBox("Manual Basic Adjustments")
        manual_layout = QGridLayout(manual_group)
        self.adjustment_sliders = {}
        adjustment_specs = [
            ("Temp", "temp", -100, 100, 0),
            ("Tint", "tint", -100, 100, 0),
            ("Exposure", "exposure", -200, 200, 0),
            ("Contrast", "contrast", -100, 100, 0),
            ("Highlights", "highlights", -100, 100, 0),
            ("Shadows", "shadows", -100, 100, 0),
            ("Whites", "whites", -100, 100, 0),
            ("Blacks", "blacks", -100, 100, 0),
            ("Vibrance", "vibrance", -100, 100, 0),
            ("Saturation", "saturation", -100, 100, 0),
            ("Clarity", "clarity", -100, 100, 0),
            ("Dehaze", "dehaze", -100, 100, 0),
            ("Sharpen", "sharpen", 0, 100, 0),
        ]
        for row, (label, key, min_val, max_val, default) in enumerate(adjustment_specs):
            manual_layout.addWidget(QLabel(label + ":"), row, 0)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(min_val, max_val)
            slider.setValue(default)
            value_label = QLabel(str(default))
            slider.valueChanged.connect(lambda value, lbl=value_label: lbl.setText(str(value)))
            manual_layout.addWidget(slider, row, 1)
            manual_layout.addWidget(value_label, row, 2)
            self.adjustment_sliders[key] = slider
        self.btn_reset_adjustments = QPushButton("Reset Manual")
        self.btn_reset_adjustments.clicked.connect(self.reset_manual_adjustments)
        manual_layout.addWidget(self.btn_reset_adjustments, len(adjustment_specs), 0, 1, 3)
        top_layout.addWidget(manual_group, 4)

        top_layout.addWidget(photo_group, 4)

        # Top-Right: Video Settings
        video_group = QGroupBox("Video Settings")
        video_layout = QGridLayout(video_group)

        self.cb_convert_videos = QCheckBox("Convert videos to MP4")
        self.cb_convert_videos.setChecked(True)
        video_layout.addWidget(self.cb_convert_videos, 0, 0, 1, 2)

        # Output Preset
        video_layout.addWidget(QLabel("Preset:"), 1, 0)
        self.cmb_video_preset = QComboBox()
        self.cmb_video_preset.addItems([
            "WhatsApp Balanced 1080p",
            "High Quality 1080p",
            "Small File 720p",
            "Original Resolution High Quality"
        ])
        video_layout.addWidget(self.cmb_video_preset, 1, 1)

        # FPS Selection
        video_layout.addWidget(QLabel("FPS limit:"), 2, 0)
        self.cmb_video_fps = QComboBox()
        self.cmb_video_fps.addItems(["Keep original", "30 fps", "60 fps"])
        video_layout.addWidget(self.cmb_video_fps, 2, 1)

        # Faststart Checkbox
        self.cb_video_faststart = QCheckBox("Enable Faststart (Streamable)")
        self.cb_video_faststart.setChecked(True)
        video_layout.addWidget(self.cb_video_faststart, 3, 0, 1, 2)

        # Video suffix
        video_layout.addWidget(QLabel("Video suffix:"), 4, 0)
        self.txt_video_suffix = QLineEdit("_WA1080p")
        video_layout.addWidget(self.txt_video_suffix, 4, 1)

        top_layout.addWidget(video_group, 4)

        # Bottom section: File Table & Logs
        bottom_widget = QSplitter(Qt.Horizontal)
        main_splitter.addWidget(bottom_widget)

        # File Queue Table Widget
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Status", "Type", "Input File", "Output File", "Size Before", "Size After", "Message"
        ])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        bottom_widget.addWidget(self.table)

        # Log Display
        log_panel = QWidget()
        log_panel_layout = QVBoxLayout(log_panel)
        log_panel_layout.setContentsMargins(0, 0, 0, 0)
        
        log_panel_layout.addWidget(QLabel("Process Log:"))
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        log_panel_layout.addWidget(self.log_display)
        
        self.btn_save_log = QPushButton("Save Log")
        self.btn_save_log.clicked.connect(self.save_log_to_file)
        log_panel_layout.addWidget(self.btn_save_log)

        bottom_widget.addWidget(log_panel)
        
        # Set splitter sizes
        main_splitter.setSizes([300, 400])
        bottom_widget.setSizes([700, 300])

        # Bottom Status Bar & Action controls
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        
        self.lbl_status = QLabel("Ready. Drag files or use Select buttons above.")
        
        status_vbox = QVBoxLayout()
        status_vbox.addWidget(self.lbl_status)
        status_vbox.addWidget(self.progress_bar)
        
        actions_layout.addLayout(status_vbox, 3)

        self.btn_start = QPushButton("Start Batch")
        self.btn_start.clicked.connect(self.start_batch)
        
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_batch)

        self.btn_open_output = QPushButton("Open Output Folder")
        self.btn_open_output.clicked.connect(self.open_output_folder)
        
        self.btn_remove_selected = QPushButton("Remove Selected")
        self.btn_remove_selected.clicked.connect(self.remove_selected_item)
        
        self.btn_clear = QPushButton("Clear Queue")
        self.btn_clear.clicked.connect(self.clear_queue)

        actions_layout.addWidget(self.btn_start)
        actions_layout.addWidget(self.btn_cancel)
        actions_layout.addWidget(self.btn_open_output)
        actions_layout.addWidget(self.btn_remove_selected)
        actions_layout.addWidget(self.btn_clear)

        main_layout.addWidget(actions_widget)

    # Drag and Drop handlers
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = []
        for url in event.mimeData().urls():
            p = url.toLocalFile()
            if p:
                paths.append(p)
        
        if paths:
            self.process_added_paths(paths)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Photo/Video Files",
            "",
            "All Supported (*.cr3 *.CR3 *.cr2 *.CR2 *.jpg *.JPG *.jpeg *.JPEG *.mov *.MOV *.mp4 *.MP4 *.m4v *.M4V *.mts *.MTS);;" +
            "Photos (*.cr3 *.CR3 *.cr2 *.CR2 *.jpg *.JPG *.jpeg *.JPEG);;" +
            "Videos (*.mov *.MOV *.mp4 *.MP4 *.m4v *.M4V *.mts *.MTS)"
        )
        if files:
            self.process_added_paths(files)

    def select_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Folder to Scan")
        if folder:
            self.process_added_paths([folder])

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.selected_output_dir = folder
            self.lbl_output_path.setText(folder)
            self.lbl_output_path.setStyleSheet("color: #000; font-weight: bold;")
            self.log(f"Output folder set to: {folder}")

    def process_added_paths(self, paths: list):
        self.lbl_status.setText("Scanning files...")
        
        # Scan files
        recursive = self.cb_recursive.isChecked()
        photos, videos = scan_paths(paths, recursive, self.selected_output_dir)
        
        # Filter existing paths to avoid duplicates
        existing_inputs = {item['input_path'] for item in self.scanned_files}
        
        added_count = 0
        
        for p in photos:
            if p not in existing_inputs:
                base_dir = self.find_base_directory(p, paths)
                self.scanned_files.append({
                    'input_path': p,
                    'base_dir': base_dir,
                    'type': 'Photo'
                })
                existing_inputs.add(p)
                added_count += 1
                
        for v in videos:
            if v not in existing_inputs:
                base_dir = self.find_base_directory(v, paths)
                self.scanned_files.append({
                    'input_path': v,
                    'base_dir': base_dir,
                    'type': 'Video'
                })
                existing_inputs.add(v)
                added_count += 1

        self.log(f"Scanned paths. Added {added_count} new file(s) to the queue.")
        self.refresh_table()
        self.lbl_status.setText(f"Queue: {len(self.scanned_files)} file(s) loaded.")

    def find_base_directory(self, file_path: str, source_paths: list) -> str:
        """Finds the root scanned folder that contains this file."""
        for p in source_paths:
            if os.path.isdir(p) and file_path.startswith(p):
                return p
        return os.path.dirname(file_path)

    def refresh_table(self):
        # We preserve the current table rows but rebuild to match queue list
        self.table.setRowCount(len(self.scanned_files))
        for idx, item in enumerate(self.scanned_files):
            input_path = item['input_path']
            file_type = item['type']
            
            # Status
            status_item = QTableWidgetItem("Pending")
            self.table.setItem(idx, 0, status_item)
            
            # Type
            type_item = QTableWidgetItem(file_type)
            self.table.setItem(idx, 1, type_item)
            
            # Input Path
            input_item = QTableWidgetItem(os.path.basename(input_path))
            input_item.setToolTip(input_path)
            self.table.setItem(idx, 2, input_item)
            
            # Output Path (empty initially)
            self.table.setItem(idx, 3, QTableWidgetItem(""))
            
            # Size Before
            try:
                sz = os.path.getsize(input_path)
                size_before_item = QTableWidgetItem(format_size(sz))
            except Exception:
                size_before_item = QTableWidgetItem("N/A")
            self.table.setItem(idx, 4, size_before_item)
            
            # Size After
            self.table.setItem(idx, 5, QTableWidgetItem(""))
            
            # Message
            self.table.setItem(idx, 6, QTableWidgetItem(""))

    def clear_queue(self):
        if self.processing_worker and self.processing_worker.isRunning():
            QMessageBox.warning(self, "Warning", "Cannot clear queue while conversion is in progress.")
            return
        self.scanned_files = []
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Queue cleared.")
        self.log("Queue cleared.")

    def reset_manual_adjustments(self):
        for slider in self.adjustment_sliders.values():
            slider.setValue(0)
        self.log("Manual photo adjustments reset.")

    def apply_photo_export_preset(self, preset: str):
        if preset == "Archive High Quality":
            self.slider_quality.setValue(95)
            self.cmb_photo_max_dimension.setCurrentText("Original")
            self.cb_photo_resolution.setChecked(True)
        elif preset == "Office Share Balanced":
            self.slider_quality.setValue(92)
            self.cmb_photo_max_dimension.setCurrentText("3000")
            self.cb_photo_resolution.setChecked(False)
        elif preset == "WhatsApp / Social":
            self.slider_quality.setValue(88)
            self.cmb_photo_max_dimension.setCurrentText("1920")
            self.cb_photo_resolution.setChecked(False)

    def get_manual_adjustments(self) -> dict:
        values = {key: slider.value() for key, slider in self.adjustment_sliders.items()}
        values["exposure"] = values.get("exposure", 0) / 100.0
        return values

    def get_settings(self) -> dict:
        return {
            'output_dir': self.selected_output_dir,
            'preserve_structure': self.cb_preserve_structure.isChecked(),
            'create_subfolders': self.cb_create_subfolders.isChecked(),
            
            # Photo settings
            'convert_photos': self.cb_convert_photos.isChecked(),
            'photo_quality': self.slider_quality.value(),
            'photo_keep_resolution': self.cb_photo_resolution.isChecked(),
            'photo_autocorrect': self.cmb_autocorrect.currentText(),
            'photo_camera_wb': self.cb_camera_wb.isChecked(),
            'photo_copy_exif': self.cb_copy_exif.isChecked(),
            'photo_correct_corner_shading': self.cb_raw_corner_shading.isChecked(),
            'raw_engine': self.cmb_raw_engine.currentText(),
            'photo_export_preset': self.cmb_photo_export_preset.currentText(),
            'photo_max_dimension': self.cmb_photo_max_dimension.currentText(),
            'photo_auto_each': self.cb_auto_each_photo.isChecked(),
            'manual_adjustments': self.get_manual_adjustments(),
            'photo_suffix': self.txt_photo_suffix.text(),

            # Video settings
            'convert_videos': self.cb_convert_videos.isChecked(),
            'video_preset': self.cmb_video_preset.currentText(),
            'video_fps': self.cmb_video_fps.currentText(),
            'video_faststart': self.cb_video_faststart.isChecked(),
            'video_suffix': self.txt_video_suffix.text()
        }

    def start_batch(self):
        if not self.scanned_files:
            QMessageBox.information(self, "RAW2Share", "No files in the queue to process.")
            return

        settings = self.get_settings()
        
        # Build worker queue by filtering depending on toggles
        worker_queue = []
        for idx, item in enumerate(self.scanned_files):
            # Reset table row status
            self.table.item(idx, 0).setText("Pending")
            self.table.item(idx, 3).setText("")
            self.table.item(idx, 5).setText("")
            self.table.item(idx, 6).setText("")

            is_photo_item = item['type'] == 'Photo'
            is_video_item = item['type'] == 'Video'
            
            should_process = False
            if is_photo_item and settings['convert_photos']:
                should_process = True
            elif is_video_item and settings['convert_videos']:
                should_process = True
            
            if should_process:
                worker_queue.append({
                    'index': idx,
                    'input_path': item['input_path'],
                    'base_dir': item['base_dir'],
                    'type': item['type']
                })
            else:
                self.table.item(idx, 0).setText("Skipped")
                self.table.item(idx, 6).setText("Skipped by user settings.")

        if not worker_queue:
            QMessageBox.information(self, "RAW2Share", "All queue files are skipped under current settings.")
            return

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_clear.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(100)

        # Create worker
        self.processing_worker = ConversionWorker(worker_queue, settings)
        self.processing_worker.file_started.connect(self.on_file_started)
        self.processing_worker.file_progress.connect(self.on_file_progress)
        self.processing_worker.file_completed.connect(self.on_file_completed)
        self.processing_worker.all_completed.connect(self.on_all_completed)
        self.processing_worker.log_message.connect(self.log)
        
        self.processing_worker.start()

    def cancel_batch(self):
        if self.processing_worker:
            self.lbl_status.setText("Cancelling... waiting for current file to complete/stop.")
            self.processing_worker.cancel()
            self.btn_cancel.setEnabled(False)

    # Worker callbacks
    def on_file_started(self, idx: int, input_path: str):
        self.table.item(idx, 0).setText("Processing")
        self.lbl_status.setText(f"Processing: {os.path.basename(input_path)}")
        self.log(f"Started processing: {os.path.basename(input_path)}")

    def on_file_progress(self, idx: int, progress: float):
        # Update current row progress if we want, or status bar
        self.table.item(idx, 6).setText(f"Progress: {progress:.1f}%")
        
        # Compute overall progress
        # For simplicity, base it on finished files + current file percentage
        total_files = len(self.processing_worker.queue)
        if total_files > 0:
            # Let's count finished files before the current one
            current_worker_idx = -1
            for queue_idx, q_item in enumerate(self.processing_worker.queue):
                if q_item['index'] == idx:
                    current_worker_idx = queue_idx
                    break
            
            if current_worker_idx >= 0:
                completed_portion = (current_worker_idx / total_files) * 100.0
                current_portion = (progress / total_files)
                overall_progress = completed_portion + current_portion
                self.progress_bar.setValue(int(overall_progress))

    def on_file_completed(self, idx: int, success: bool, message: str, size_before: int, size_after: int, output_path: str):
        if success:
            self.table.item(idx, 0).setText("Success")
            self.table.item(idx, 3).setText(os.path.basename(output_path))
            self.table.item(idx, 3).setToolTip(output_path)
            self.table.item(idx, 5).setText(format_size(size_after))
        else:
            self.table.item(idx, 0).setText("Failed")
            
        self.table.item(idx, 4).setText(format_size(size_before))
        self.table.item(idx, 6).setText(message)

    def on_all_completed(self):
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Ready.")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_clear.setEnabled(True)
        
        QMessageBox.information(self, "RAW2Share", "Processing completed!")
        self.processing_worker = None

    def open_output_folder(self):
        # Open output dir
        out_dir = self.selected_output_dir
        if not out_dir:
            # Default to checking if we processed something and have a fallback
            if self.scanned_files:
                out_dir = os.path.join(os.path.dirname(self.scanned_files[0]['input_path']), "RAW2Share_Output")
            else:
                out_dir = os.getcwd()

        if not os.path.exists(out_dir):
            try:
                os.makedirs(out_dir, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Could not create output folder: {str(e)}")
                return

        try:
            if sys.platform == 'win32':
                os.startfile(out_dir)
            elif sys.platform == 'darwin':
                subprocess.run(['open', out_dir])
            else:
                subprocess.run(['xdg-open', out_dir])
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open output folder: {str(e)}")

    def remove_selected_item(self):
        if self.processing_worker and self.processing_worker.isRunning():
            QMessageBox.warning(self, "Warning", "Cannot modify queue while conversion is in progress.")
            return
            
        selected_ranges = self.table.selectedRanges()
        if not selected_ranges:
            QMessageBox.information(self, "RAW2Share", "Please select a row in the table to remove.")
            return

        # Find all selected row indices
        rows_to_remove = set()
        for r in selected_ranges:
            for row in range(r.topRow(), r.bottomRow() + 1):
                rows_to_remove.add(row)
                
        # Sort rows in descending order to avoid shift issues during deletion
        sorted_rows = sorted(list(rows_to_remove), reverse=True)
        
        for r_idx in sorted_rows:
            if 0 <= r_idx < len(self.scanned_files):
                self.scanned_files.pop(r_idx)
                
        # Refresh indexing of remaining files
        for new_idx, item in enumerate(self.scanned_files):
            # We don't have item['index'] directly used inside self.scanned_files but let's check
            pass
            
        self.refresh_table()
        self.lbl_status.setText(f"Queue: {len(self.scanned_files)} file(s) loaded.")
        self.log(f"Removed {len(sorted_rows)} item(s) from the queue.")

    def save_log_to_file(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Log File",
            os.path.join(os.path.expanduser("~"), "RAW2Share_log.txt"),
            "Text Files (*.txt)"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("\n".join(self.logs))
                QMessageBox.information(self, "RAW2Share", "Log saved successfully.")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Could not save log file: {str(e)}")




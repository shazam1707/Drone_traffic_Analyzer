from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QProgressBar, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox,
    QStatusBar, QSplitter, QFrame
)
from PyQt6.QtCore import Qt, QThread
from PyQt6.QtGui import QImage, QPixmap
import numpy as np
from core.worker import VideoWorker


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Drone Traffic Analyzer")
        self.setMinimumSize(1100, 700)
        self._thread = None
        self._worker = None
        self._report_path = None
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # ── Left panel: video display ──────────────────────────
        left = QVBoxLayout()

        self.video_label = QLabel("No video loaded")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet(
            "background:#111; color:#888; border-radius:8px;"
        )
        left.addWidget(self.video_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        left.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        self.btn_upload   = QPushButton("Upload video")
        self.btn_cancel   = QPushButton("Cancel")
        self.btn_download = QPushButton("Download report")
        self.btn_cancel.setEnabled(False)
        self.btn_download.setEnabled(False)

        self.btn_upload.clicked.connect(self._on_upload)
        self.btn_cancel.clicked.connect(self._on_cancel)
        self.btn_download.clicked.connect(self._on_download)

        for b in [self.btn_upload, self.btn_cancel, self.btn_download]:
            btn_row.addWidget(b)
        left.addLayout(btn_row)

        # ── Right panel: summary + detection log ───────────────
        right = QVBoxLayout()

        self.summary_label = QLabel("Summary will appear here after processing.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet(
            "padding:12px; background:#1a1a2e; border-radius:8px; color:#eee;"
        )
        right.addWidget(self.summary_label)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            ["Track ID", "Class", "First frame", "Timestamp (s)"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.table)

        root.addLayout(left, 3)
        root.addLayout(right, 2)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready — upload a drone video to begin.")

    # ── Signal slots ───────────────────────────────────────────

    def _on_upload(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select drone video", "", "Video files (*.mp4 *.avi *.mov)"
        )
        if not path:
            return
        self._start_processing(path)

    def _start_processing(self, path: str):
        # Reset UI
        self.table.setRowCount(0)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.btn_upload.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_download.setEnabled(False)
        self._report_path = None

        # Create thread + worker
        self._thread = QThread()
        self._worker = VideoWorker(path)
        self._worker.moveToThread(self._thread)

        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.progress_updated.connect(self.progress_bar.setValue)
        self._worker.frame_ready.connect(self._update_frame)
        self._worker.vehicle_detected.connect(self._add_table_row)
        self._worker.status_message.connect(self.status_bar.showMessage)
        self._worker.finished.connect(self._on_finished)
        self._worker.error_occurred.connect(self._on_error)

        # Clean up thread when done
        self._worker.finished.connect(self._thread.quit)
        self._worker.error_occurred.connect(self._thread.quit)
        self._thread.finished.connect(self._thread.deleteLater)

        self._thread.start()

    def _on_cancel(self):
        if self._worker:
            self._worker.stop()
        self.btn_cancel.setEnabled(False)
        self.btn_upload.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage("Processing cancelled.")

    def _on_download(self):
        if not self._report_path:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save report", "vehicle_report.xlsx",
            "Excel files (*.xlsx)"
        )
        if dest:
            import shutil
            shutil.copy(self._report_path, dest)
            self.status_bar.showMessage(f"Report saved to {dest}")

    def _update_frame(self, frame: np.ndarray):
        """Convert OpenCV BGR frame → QPixmap and display."""
        h, w, ch = frame.shape
        rgb = frame[:, :, ::-1].copy()          # BGR → RGB
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pix  = QPixmap.fromImage(qimg).scaled(
            self.video_label.width(),
            self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pix)

    def _add_table_row(self, record: dict):
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(str(record["tracker_id"])))
        self.table.setItem(row, 1, QTableWidgetItem(record["class"]))
        self.table.setItem(row, 2, QTableWidgetItem(str(record["first_frame"])))
        self.table.setItem(row, 3, QTableWidgetItem(str(record["timestamp_s"])))
        self.table.scrollToBottom()

    def _on_finished(self, summary: dict):
        self._report_path = summary["report_path"]
        self.progress_bar.setValue(100)
        self.btn_upload.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_download.setEnabled(True)

        from collections import Counter
        types = Counter(v["class"] for v in summary["vehicles"])
        breakdown = "  |  ".join(f"{k}: {v}" for k, v in types.items())
        self.summary_label.setText(
            f"Total unique vehicles: {summary['total']}\n"
            f"{breakdown}\n"
            f"Processing time: {summary['duration_s']}s"
        )
        self.status_bar.showMessage("Processing complete — report ready to download.")

    def _on_error(self, message: str):
        self.btn_upload.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Processing error", message)
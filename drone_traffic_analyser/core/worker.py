# core/worker.py  (updated run method only — rest stays the same)

from PyQt6.QtCore import QObject, pyqtSignal
import numpy as np
from core.pipeline import process_video
from core.reporter import generate_report
import os


class VideoWorker(QObject):
    progress_updated = pyqtSignal(int)
    frame_ready      = pyqtSignal(np.ndarray)
    vehicle_detected = pyqtSignal(dict)
    status_message   = pyqtSignal(str)
    finished         = pyqtSignal(dict)
    error_occurred   = pyqtSignal(str)

    def __init__(self, video_path: str):
        super().__init__()
        self.video_path = video_path
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            base        = os.path.splitext(self.video_path)[0]
            output_vid  = base + "_annotated.mp4"
            output_rep  = base + "_report.xlsx"

            summary = process_video(
                video_path        = self.video_path,
                output_video_path = output_vid,
                progress_callback = self.progress_updated.emit,
                frame_callback    = self.frame_ready.emit,
                vehicle_callback  = self.vehicle_detected.emit,
                status_callback   = self.status_message.emit,
                stop_flag         = lambda: self._stop,
            )

            generate_report(
                vehicle_log = summary["vehicles"],
                duration    = summary["duration_s"],
                output_path = output_rep,
            )

            summary["report_path"]        = output_rep
            summary["annotated_video_path"] = output_vid
            self.finished.emit(summary)

        except Exception as e:
            self.error_occurred.emit(str(e))
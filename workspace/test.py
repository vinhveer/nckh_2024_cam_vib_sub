import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QFileDialog, 
                             QPushButton, QMessageBox, QVBoxLayout, QHBoxLayout, 
                             QWidget, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from pypylon import pylon

class BaslerCameraROITracker(QMainWindow):
    def __init__(self):
        super().__init__()

        # GUI components
        self.init_ui()

        # Basler Camera Setup
        self.camera = None
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

        # State variables
        self.current_image = None
        self.template = None
        self.roi_start = None
        self.roi_end = None
        self.is_selecting = False
        self.is_tracking = False

        # Tracking timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.track_continuously)

    def init_ui(self):
        """Initialize the GUI layout."""
        self.setWindowTitle("Basler Camera ROI Tracker")
        
        # Widgets
        self.label_display = QLabel("No image captured")
        self.label_display.setAlignment(Qt.AlignCenter)

        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)

        self.btn_capture = QPushButton("Capture Image")
        self.btn_capture.clicked.connect(self.capture_first_image)

        self.btn_select_roi = QPushButton("Select ROI")
        self.btn_select_roi.clicked.connect(self.start_roi_selection)

        self.btn_start_tracking = QPushButton("Start Tracking")
        self.btn_start_tracking.clicked.connect(self.start_tracking)

        self.btn_stop_tracking = QPushButton("Stop Tracking")
        self.btn_stop_tracking.clicked.connect(self.stop_tracking)

        # Layout
        layout_buttons = QHBoxLayout()
        layout_buttons.addWidget(self.btn_capture)
        layout_buttons.addWidget(self.btn_select_roi)
        layout_buttons.addWidget(self.btn_start_tracking)
        layout_buttons.addWidget(self.btn_stop_tracking)

        layout_main = QVBoxLayout()
        layout_main.addWidget(self.label_display)
        layout_main.addLayout(layout_buttons)
        layout_main.addWidget(self.text_info)

        container = QWidget()
        container.setLayout(layout_main)
        self.setCentralWidget(container)

    def capture_first_image(self):
        """Capture the first image from the Basler camera."""
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

            if grab_result.GrabSucceeded():
                image = self.converter.Convert(grab_result)
                self.current_image = image.GetArray()
                self.display_image(self.current_image)
                self.text_info.append("Image captured successfully.")
            else:
                QMessageBox.critical(self, "Error", "Failed to grab image from camera.")

            grab_result.Release()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Camera error: {e}")

    def display_image(self, image):
        """Display an image in the QLabel widget."""
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        q_image = QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        self.label_display.setPixmap(pixmap.scaled(self.label_display.size(), Qt.KeepAspectRatio))

    def start_roi_selection(self):
        """Initiate ROI selection by mouse interaction."""
        if self.current_image is None:
            QMessageBox.warning(self, "Error", "Please capture an image first.")
            return

        self.text_info.append("Draw a rectangle to select ROI.")
        self.label_display.mousePressEvent = self.begin_roi_selection
        self.label_display.mouseReleaseEvent = self.end_roi_selection
        self.label_display.mouseMoveEvent = self.update_roi_selection

        self.is_selecting = True

    def begin_roi_selection(self, event):
        if event.button() == Qt.LeftButton:
            self.roi_start = (event.x(), event.y())

    def update_roi_selection(self, event):
        if self.is_selecting and self.roi_start is not None:
            self.roi_end = (event.x(), event.y())

    def end_roi_selection(self, event):
        if not self.is_selecting or self.roi_start is None or self.roi_end is None:
            return

        x1, y1 = self.roi_start
        x2, y2 = self.roi_end
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])

        self.template = self.current_image[y1:y2, x1:x2]
        self.text_info.append("ROI selected successfully.")
        self.is_selecting = False

    def start_tracking(self):
        """Start continuous tracking."""
        if self.template is not None:
            self.is_tracking = True
            self.timer.start(30)  # Update every 30 ms
        else:
            QMessageBox.warning(self, "Error", "Please capture an image and select ROI first.")

    def stop_tracking(self):
        """Stop the tracking process."""
        self.is_tracking = False
        self.timer.stop()
        self.text_info.append("Tracking stopped.")

    def track_continuously(self):
        """Continuously track the selected ROI in the video feed."""
        if not self.is_tracking:
            return

        grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            frame = self.converter.Convert(grab_result).GetArray()

            result = self.template_match(frame, self.template)
            if result is not None:
                x, y, w, h = result
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self.display_image(frame)
        grab_result.Release()

    @staticmethod
    def template_match(frame, template):
        """Perform template matching to find ROI in the current frame."""
        res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.8:
            h, w = template.shape[:2]
            return max_loc[0], max_loc[1], w, h
        return None

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BaslerCameraROITracker()
    window.show()
    sys.exit(app.exec_())

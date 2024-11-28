import sys
import cv2
import time
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, 
                             QMessageBox, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from pypylon import pylon
import matplotlib.pyplot as plt


class EnhancedROITracker:
    @staticmethod
    def multi_method_tracking(image, template, initial_x, initial_y, initial_search_area=50):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        
        feature_result = EnhancedROITracker._feature_match(gray_image, gray_template, initial_x, initial_y, initial_search_area)
        
        if feature_result is None:
            template_result = EnhancedROITracker._template_match(gray_image, gray_template, initial_x, initial_y, initial_search_area)
            return template_result
        
        return feature_result

    @staticmethod
    def _template_match(image, template, initial_x, initial_y, initial_search_area):
        h, w = template.shape[:2]
        search_area = initial_search_area
        
        while search_area <= max(image.shape[:2]):
            x1 = max(0, initial_x - search_area)
            y1 = max(0, initial_y - search_area)
            x2 = min(image.shape[1], initial_x + search_area)
            y2 = min(image.shape[0], initial_y + search_area)
            
            search_region = image[y1:y2, x1:x2]
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.65:
                found_x = x1 + max_loc[0] + w // 2
                found_y = y1 + max_loc[1] + h // 2
                return found_x, found_y, max_val
            
            search_area *= 2
        
        return None

    @staticmethod
    def _feature_match(image, template, initial_x, initial_y, initial_search_area):
        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(image, None)
        
        if des1 is None or des2 is None:
            return None
        
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        
        good_matches = [
            m for m in matches 
            if abs(kp2[m.trainIdx].pt[0] - initial_x) <= initial_search_area and
               abs(kp2[m.trainIdx].pt[1] - initial_y) <= initial_search_area
        ]
        
        if good_matches:
            best_match = good_matches[0]
            found_x, found_y = kp2[best_match.trainIdx].pt
            confidence = max(0, 1 - (best_match.distance / 500))
            return int(found_x), int(found_y), confidence
        
        return None


class BaslerCameraROITracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracking_data = []
        self.frame_count = 0

        self.setWindowTitle("Basler Camera ROI Tracking")
        self.setGeometry(100, 100, 800, 600)
        
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            self.camera.Open()

            self.camera.Width.Value = 1024  # Đặt chiều rộng hình ảnh
            self.camera.Height.Value = 1024  # Đặt chiều cao hình ảnh

            self.camera.AcquisitionFrameRateEnable.Value = True
            self.camera.AcquisitionFrameRate.Value = 120.0
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        except Exception as e:
            QMessageBox.critical(self, "Camera Initialization Error", str(e))
            sys.exit(1)

        self.template = None
        self.roi_rect = None
        self.is_tracking = False

        # Layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Thêm nhãn hiển thị số khung hình
        self.frame_count_label = QLabel("Frames: 0")
        layout.addWidget(self.frame_count_label)

        # FPS display
        self.label_fps = QLabel("FPS: 0")
        layout.addWidget(self.label_fps)

        # Tracking result display
        self.label_tracked = QLabel("Tracked ROI")
        self.label_tracked.setMinimumSize(800, 600)
        self.label_tracked.setAlignment(Qt.AlignCenter)
        self.label_tracked.setStyleSheet("border: 2px solid red;")
        layout.addWidget(self.label_tracked)

        # Button layout
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.capture_first_image_button = QPushButton("Capture & Select ROI")
        self.start_tracking_button = QPushButton("Start Tracking")
        self.stop_tracking_button = QPushButton("Stop Tracking")

        button_layout.addWidget(self.capture_first_image_button)
        button_layout.addWidget(self.start_tracking_button)
        button_layout.addWidget(self.stop_tracking_button)

        self.capture_first_image_button.clicked.connect(self.capture_and_select_roi)
        self.start_tracking_button.clicked.connect(self.start_tracking)
        self.stop_tracking_button.clicked.connect(self.stop_tracking)

    def display_image(self, image):
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.label_tracked.setPixmap(pixmap.scaled(self.label_tracked.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def capture_and_select_roi(self):
        try:
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                first_image = grabResult.Array
                roi = cv2.selectROI("Select ROI", first_image)
                cv2.destroyAllWindows()

                x, y, w, h = roi
                if w > 0 and h > 0:
                    self.template = first_image[y:y+h, x:x+w]
                    self.roi_rect = (x, y, x + w, y + h)
                    QMessageBox.information(self, "ROI Selected", f"ROI: {self.roi_rect}")
                else:
                    QMessageBox.warning(self, "Error", "Invalid ROI selection")
            else:
                QMessageBox.warning(self, "Capture Error", "Failed to capture image")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def start_tracking(self):
        if self.template is None:
            QMessageBox.warning(self, "Error", "Please capture and select ROI first")
            return

        self.is_tracking = True
        self.track()

    def track(self):
        while self.is_tracking:
            try:
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    current_frame = grabResult.Array
                    x1, y1, x2, y2 = self.roi_rect
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    result = EnhancedROITracker.multi_method_tracking(current_frame, self.template, center_x, center_y, initial_search_area=100)

                    if result:
                        found_x, found_y, confidence = result
                        w = x2 - x1
                        h = y2 - y1
                        new_x1 = found_x - w // 2
                        new_y1 = found_y - h // 2
                        new_x2 = new_x1 + w
                        new_y2 = new_y1 + h
                        self.roi_rect = (new_x1, new_y1, new_x2, new_y2)

                        # Lưu dữ liệu tracking
                        self.tracking_data.append((found_x, found_y, confidence))

                        cv2.rectangle(current_frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
                        self.display_image(current_frame)

                        resulting_fps = self.camera.ResultingFrameRate.Value
                        self.label_fps.setText(f"FPS: {resulting_fps:.2f}")

                    # Cập nhật số khung hình
                    self.frame_count += 1
                    self.frame_count_label.setText(f"Frames: {self.frame_count}")

                QApplication.processEvents()
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Tracking Error: {str(e)}")
                self.stop_tracking()

    def stop_tracking(self):
        self.is_tracking = False

def main():
    app = QApplication(sys.argv)
    window = BaslerCameraROITracker()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

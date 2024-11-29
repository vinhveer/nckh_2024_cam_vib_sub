import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QVBoxLayout, QWidget, QMessageBox, QDialog, QComboBox, QDialogButtonBox)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from pypylon import pylon

class BaslerFeatureTracker(QMainWindow):
    def __init__(self):
        super().__init__()

        # Camera selection dialog
        camera_dialog = self._create_camera_selection_dialog()
        if camera_dialog.exec_() == QDialog.Accepted:
            selected_serial = camera_dialog.get_selected_camera_serial()
        else:
            sys.exit(1)

        # Initialize camera
        self._init_camera(selected_serial)
        self._setup_ui()
        
        # Tracking variables
        self.feature_points = []

        # Timer for updating frames
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.capture_frame)
        self.timer.start(30)  # Update every 30ms

        # Khai báo thêm thuộc tính để lưu trữ điểm đặc trưng
        self.feature_points = []
        self.current_frame = None
        self.template = None

        # Thêm nút để chạy thuật toán
        self.track_btn.clicked.connect(self.start_tracking)

    def _create_camera_selection_dialog(self):
        dialog = QDialog()
        dialog.setWindowTitle("Choose Basler Camera")
        layout = QVBoxLayout()
        
        # Camera selection dropdown
        camera_combo = QComboBox()
        camera_list = pylon.TlFactory.GetInstance().EnumerateDevices()
        
        for camera in camera_list:
            camera_info = f"{camera.GetSerialNumber()} - {camera.GetModelName()}"
            camera_combo.addItem(camera_info, userData=camera.GetSerialNumber())
        
        layout.addWidget(camera_combo)
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        dialog.get_selected_camera_serial = lambda: camera_combo.currentData()
        
        return dialog

    def _init_camera(self, serial_number):
        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            
            selected_device = next((device for device in devices if device.GetSerialNumber() == serial_number), None)
            
            if selected_device is None:
                raise RuntimeError("Selected camera not found")

            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(selected_device))
            self.camera.Open()

            # Basic camera settings
            self.camera.Width.Value = 600
            self.camera.Height.Value = 600
            self.camera.ExposureTime.SetValue(20000)
            self.camera.AcquisitionFrameRateEnable.Value = True
            self.camera.AcquisitionFrameRate.Value = 120.0
            
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        except Exception as e:
            QMessageBox.critical(self, "Camera Initialization Error", str(e))
            sys.exit(1)

    def _setup_ui(self):
        self.setWindowTitle("Basler Feature Tracking")
        self.setGeometry(100, 100, 800, 600)
        
        central_widget = QWidget()
        layout = QVBoxLayout()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
        
        self.image_label = QLabel("Click to select feature points")
        self.image_label.setMinimumSize(640, 480)
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label)
        
        self.coordinates_label = QLabel("Coordinates will appear here")
        self.coordinates_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.coordinates_label)
        
        self.track_btn = QPushButton("Start Tracking")
        self.track_btn.clicked.connect(self.start_tracking)
        layout.addWidget(self.track_btn)
        
        self.show()

    def capture_frame(self):
        try:
            grab_result = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                self.current_frame = grab_result.Array
                self.display_image(self.current_frame)
                
                # Enable point selection only after frame capture
                self.image_label.mousePressEvent = self.select_feature_point
        except Exception as e:
            QMessageBox.critical(self, "Capture Error", str(e))

    def select_feature_point(self, event):
        x = event.pos().x()
        y = event.pos().y()
        
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        img_width = self.current_frame.shape[1]
        img_height = self.current_frame.shape[0]
        
        # Calculate coordinates on the original image
        scaled_x = int(x * img_width / label_width)
        scaled_y = int(y * img_height / label_height)
        self.feature_points.append((scaled_x, scaled_y))
        
        # Redraw all points
        display_frame = self.current_frame.copy()
        for point in self.feature_points:
            cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
        
        # Display updated image with points
        self.display_image(display_frame)
        
        # Update coordinates list
        self.update_coordinates_display()
        
        if len(self.feature_points) >= 5:
            self.image_label.mousePressEvent = None
            QMessageBox.information(self, "Points Selected", "5 feature points selected. Click 'Start Tracking'.")

    def update_coordinates_display(self):
        """Update realtime coordinates in QLabel."""
        coordinates_text = "\n".join([f"Point {i+1}: ({x}, {y})" for i, (x, y) in enumerate(self.feature_points)])
        self.coordinates_label.setText(coordinates_text)

    def display_image(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image).scaled(
            self.image_label.size(), 
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        
        self.image_label.setPixmap(pixmap)

    def start_tracking(self):
        if len(self.feature_points) < 5:
            QMessageBox.warning(self, "Warning", "Please select at least 5 feature points.")
            return

        if self.current_frame is None:
            QMessageBox.warning(self, "Warning", "No frame captured for tracking.")
            return

        # Sử dụng một điểm đặc trưng để làm ví dụ
        feature_x, feature_y = self.feature_points[0]  # Chọn điểm đầu tiên làm ví dụ
        search_area = 50  # Kích thước vùng tìm kiếm ban đầu
        
        # Template từ điểm đặc trưng
        try:
            h, w = self.current_frame.shape[:2]
            x1 = max(0, int(feature_x - search_area // 2))
            y1 = max(0, int(feature_y - search_area // 2))
            x2 = min(w, int(feature_x + search_area // 2))
            y2 = min(h, int(feature_y + search_area // 2))
            
            self.template = self.current_frame[y1:y2, x1:x2]
            if self.template.size == 0:
                QMessageBox.warning(self, "Warning", "Template region is empty.")
                return

            # Chạy thuật toán
            result = self._feature_match(self.current_frame, self.template, feature_x, feature_y, search_area)
            if result:
                found_x, found_y, confidence = result
                QMessageBox.information(
                    self, 
                    "Feature Match Result", 
                    f"Feature found at ({found_x}, {found_y}) with confidence {confidence}."
                )
            else:
                QMessageBox.warning(self, "No Match", "Feature matching failed.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    @staticmethod
    def _feature_match(image, template, initial_x, initial_y, initial_search_area):
        try:
            # Ensure parameters are float
            initial_x, initial_y = float(initial_x), float(initial_y)
            initial_search_area = float(initial_search_area)

            # Validate image and template sizes
            if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                template = cv2.resize(template, (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_AREA)

            # Detectors configuration
            detectors = [
                cv2.ORB_create(nfeatures=500),
                cv2.AKAZE_create()
            ]

            for detector in detectors:
                try:
                    # Detect keypoints and descriptors
                    kp1, des1 = detector.detectAndCompute(template, None)
                    kp2, des2 = detector.detectAndCompute(image, None)

                    # Skip if insufficient keypoints
                    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
                        continue

                    # Brute Force Matcher
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    
                    # Filter matches near search area
                    good_matches = [
                        m for m in matches
                        if abs(kp2[m.trainIdx].pt[0] - initial_x) <= initial_search_area and
                        abs(kp2[m.trainIdx].pt[1] - initial_y) <= initial_search_area
                    ]

                    if good_matches:
                        # Best match selection
                        good_matches = sorted(good_matches, key=lambda x: x.distance)
                        best_match = good_matches[0]
                        
                        # Match coordinates
                        match_x = float(kp2[best_match.trainIdx].pt[0])
                        match_y = float(kp2[best_match.trainIdx].pt[1])

                        # Create dummy result for subpixel refinement
                        result = np.zeros((3,3), dtype=np.float32)
                        result[1,1] = 1.0  # Center point
                        
                        # Subpixel refinement
                        sub_x, sub_y = BaslerFeatureTracker._subpixel_refinement(result, (1,1))
                        
                        # Apply subpixel offset
                        found_x = round(match_x + sub_x, 8)
                        found_y = round(match_y + sub_y, 8)
                        
                        # Confidence calculation
                        confidence = round(max(0, 1 - (float(best_match.distance) / 300)), 8)
                        
                        print("Update by feature match")

                        return found_x, found_y, confidence

                except Exception as detector_error:
                    print(f"Detector error: {detector_error}")
                    return None

        except Exception as e:
            print(f"Feature match error: {e}")
            return None

    @staticmethod
    def _subpixel_refinement(result, max_loc):
        x, y = max_loc
        h, w = result.shape

        # Gaussian Fitting
        try:
            dx = [
                result[y, x - 1] if x > 0 else 0,
                result[y, x],
                result[y, x + 1] if x < w - 1 else 0
            ]
            dy = [
                result[y - 1, x] if y > 0 else 0,
                result[y, x],
                result[y + 1, x] if y < h - 1 else 0
            ]

            offset_x_gaussian = (dx[2] - dx[0]) / (2 * (2 * dx[1] - dx[0] - dx[2]) + 1e-8)
            offset_y_gaussian = (dy[2] - dy[0]) / (2 * (2 * dy[1] - dy[0] - dy[2]) + 1e-8)
        except Exception:
            offset_x_gaussian, offset_y_gaussian = 0, 0

        roi_size = 3
        roi_x_start = max(x - roi_size // 2, 0)
        roi_y_start = max(y - roi_size // 2, 0)
        roi_x_end = min(x + roi_size // 2 + 1, w)
        roi_y_end = min(y + roi_size // 2 + 1, h)

        roi = result[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        moments = cv2.moments(roi)
        if moments["m00"] != 0:
            offset_x_moments = moments["m10"] / moments["m00"] - x
            offset_y_moments = moments["m01"] / moments["m00"] - y
        else:
            offset_x_moments, offset_y_moments = 0, 0

        weighted_sum = np.sum(roi)
        if weighted_sum > 0:
            grid_x, grid_y = np.meshgrid(
                np.arange(roi_x_start, roi_x_end),
                np.arange(roi_y_start, roi_y_end)
            )
            offset_x_weighted = (np.sum(grid_x * roi) / weighted_sum) - x
            offset_y_weighted = (np.sum(grid_y * roi) / weighted_sum) - y
        else:
            offset_x_weighted, offset_y_weighted = 0, 0

        if abs(offset_x_gaussian) <= 1 and abs(offset_y_gaussian) <= 1:
            return offset_x_gaussian, offset_y_gaussian
        elif abs(offset_x_moments) <= 1 and abs(offset_y_moments) <= 1:
            return offset_x_moments, offset_y_moments
        else:
            return offset_x_weighted, offset_y_weighted

def main():
    app = QApplication(sys.argv)
    tracker = BaslerFeatureTracker()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

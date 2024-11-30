import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, 
                             QMessageBox, QVBoxLayout, QHBoxLayout, QWidget, QDialog, QComboBox, QDialogButtonBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from pypylon import pylon
import matplotlib.pyplot as plt

from camera_selection_dialog import CameraSelectionDialog
from enhanced_roi_tracker import EnhancedROITracker


class BaslerCameraROITracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracking_data = []
        self.frame_count = 0
        self.setWindowTitle("Tracking ROI")
        
        # Camera selection dialog
        camera_dialog = CameraSelectionDialog(self)
        if camera_dialog.exec_() == QDialog.Accepted:
            selected_serial = camera_dialog.get_selected_camera_serial()
        else:
            QMessageBox.critical(self, "Error", "No camera selected. Exiting.")
            sys.exit(1)

        # Initialize camera with selected serial number
        try:
            # Tạo đối tượng camera với serial number đã chọn
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            
            # Tìm camera với serial number phù hợp
            selected_device = None
            for device in devices:
                if device.GetSerialNumber() == selected_serial:
                    selected_device = device
                    break
            
            if selected_device is None:
                raise RuntimeError("Selected camera not found")

            # Mở camera được chọn
            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(selected_device))
            self.camera.Open()

            # Cài đặt thông số camera
            self.camera.Width.Value = 600  # Đặt chiều rộng hình ảnh
            self.camera.Height.Value = 600  # Đặt chiều cao hình ảnh

            self.camera.ExposureTime.SetValue(20000)  # Đặt thời gian mở

            # Đặt tốc độ khung hình
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
        self.tracked_result = QLabel("Tracked ROI")
        layout.addWidget(self.tracked_result)

        # Tracking result display
        self.label_tracked = QLabel("Tracked ROI")
        self.label_tracked.setMinimumSize(800, 600)
        self.label_tracked.setAlignment(Qt.AlignCenter)
        self.label_tracked.setStyleSheet("border: 2px solid red;")
        layout.addWidget(self.label_tracked)

        # Button layout
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        self.capture_first_image_button = QPushButton("Select ROI")
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
                
                # Create a window with size scaled to half of the original image
                h, w = first_image.shape[:2]
                scaled_w, scaled_h = w // 2, h // 2
                
                # Resize the image for display
                scaled_image = cv2.resize(first_image, (scaled_w, scaled_h))
                
                # Select ROI on the scaled image
                roi = cv2.selectROI("Select ROI", scaled_image)
                cv2.destroyAllWindows()

                # Adjust ROI coordinates to match original image scale
                x, y, w, h = roi
                original_x = int(x * 2)
                original_y = int(y * 2)
                original_w = int(w * 2)
                original_h = int(h * 2)

                if original_w > 0 and original_h > 0:
                    self.template = first_image[original_y:original_y+original_h, original_x:original_x+original_w]
                    self.roi_rect = (original_x, original_y, original_x + original_w, original_y + original_h)
                    QMessageBox.information(self, "ROI Selected", f"ROI: {self.roi_rect}")
                else:
                    QMessageBox.warning(self, "Error", "Invalid ROI selection")
            else:
                QMessageBox.warning(self, "Capture Error", "Failed to capture image")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def start_tracking(self):
        # Kiểm tra điều kiện ban đầu
        if self.template is None:
            QMessageBox.warning(self, "Error", "Please capture and select ROI first")
            return

        # Tạo dialog chọn method
        dialog = QDialog(self)
        dialog.setGeometry(200, 200, 300, 100)
        dialog.setWindowTitle("Select Tracking Method")
        
        # Dropdown chọn method
        method_selector = QComboBox(dialog)
        method_selector.addItems(["auto", "feature_match", "template_match"])
        
        # Nút OK
        ok_button = QPushButton("OK", dialog)
        ok_button.clicked.connect(dialog.accept)
        
        # Layout cho dialog
        layout = QVBoxLayout()
        layout.addWidget(method_selector)
        layout.addWidget(ok_button)
        dialog.setLayout(layout)

        # Hiển thị dialog
        if dialog.exec_() == QDialog.Accepted:
            selected_method = method_selector.currentText()  # Lấy method được chọn
            self.is_tracking = True
            self.track(method=selected_method)  # Gọi hàm track với phương pháp đã chọn
        else:
            QMessageBox.information(self, "Info", "Tracking cancelled")

    def track(self, method="auto"):
        prev_x, prev_y = None, None

        while self.is_tracking:
            try:
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    current_frame = grabResult.Array
                    x1, y1, x2, y2 = self.roi_rect
                    center_x = round(float(x1 + x2) / 2, 8)
                    center_y = round(float(y1 + y2) / 2, 8)

                    result = EnhancedROITracker.multi_method_tracking(current_frame, self.template, center_x, center_y, initial_search_area=100, method=method)

                    if result:
                        found_x, found_y, confidence = result
                        w = x2 - x1
                        h = y2 - y1
                        new_x1 = round(found_x - w / 2, 8)
                        new_y1 = round(found_y - h / 2, 8)
                        new_x2 = round(new_x1 + w, 8)
                        new_y2 = round(new_y1 + h, 8)
                        self.roi_rect = (new_x1, new_y1, new_x2, new_y2)

                        if prev_x is not None and prev_y is not None:
                            dx = round(found_x - prev_x, 8)
                            dy = round(found_y - prev_y, 8)
                        else:
                            dx = dy = 0.0

                        self.tracking_data.append((found_x, found_y, confidence, dx, dy))
                        prev_x, prev_y = found_x, found_y

                        if len(current_frame.shape) == 2:
                            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)

                        self.tracked_result.setText(f"Tracked ROI: x = {found_x:.8f}, y = {found_y:.8f}, Confidence = {confidence:.8f}")
                        
                        cv2.rectangle(current_frame, (int(new_x1), int(new_y1)), (int(new_x2), int(new_y2)), (0, 255, 0), 2)
                        self.display_image(current_frame)

                        resulting_fps = round(self.camera.ResultingFrameRate.Value, 8)
                        self.label_fps.setText(f"FPS: {resulting_fps:.8f}")

                    self.frame_count += 1
                    self.frame_count_label.setText(f"Frames: {self.frame_count}")

                    QApplication.processEvents()
                else:
                    QMessageBox.warning(self, "Error", "Capture Error: Failed to grab image")
                    self.stop_tracking()

            except Exception as e:
                QMessageBox.critical(self, "Error", str(e))
                self.stop_tracking()

    def stop_tracking(self):
        self.is_tracking = False
        self.show_plot()

    def show_plot(self):
        if not self.tracking_data:
            QMessageBox.information(self, "No Data", "No tracking data to display.")
            return
        
        # Extract data
        x_coords = [data[0] for data in self.tracking_data]
        y_coords = [data[1] for data in self.tracking_data]
        confidence = [data[2] for data in self.tracking_data]
        dx_data = [data[3] for data in self.tracking_data]
        dy_data = [data[4] for data in self.tracking_data]
        
        # Sampling frequency (assumed)
        Fs = 240  # Hz
        N = len(confidence)  # Number of data points
        t = np.linspace(0, N/Fs, N)  # Create time vector
        
        # Create figure with 2x2 grid
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Trajectory plot (top left)
        axs[0, 0].plot(x_coords, y_coords, marker='o', label="Trajectory")
        axs[0, 0].set_title("ROI Trajectory")
        axs[0, 0].set_xlabel("X")
        axs[0, 0].set_ylabel("Y")
        axs[0, 0].grid()
        axs[0, 0].legend()
        
        # Confidence plot (top right)
        axs[0, 1].plot(t, confidence, label="Confidence", color='orange')
        axs[0, 1].set_title("Confidence vs Time")
        axs[0, 1].set_xlabel("Time (s)")
        axs[0, 1].set_ylabel("Confidence")
        axs[0, 1].grid()
        axs[0, 1].legend()
        
        # dx plot (bottom left)
        axs[1, 0].plot(dx_data, label="dx", color='blue')
        axs[1, 0].set_title("dx vs Time")
        axs[1, 0].set_xlabel("Frame")
        axs[1, 0].set_ylabel("dx")
        axs[1, 0].grid()
        axs[1, 0].legend()
        
        # dy plot (bottom right)
        axs[1, 1].plot(dy_data, label="dy", color='red')
        axs[1, 1].set_title("dy vs Time")
        axs[1, 1].set_xlabel("Frame")
        axs[1, 1].set_ylabel("dy")
        axs[1, 1].grid()
        axs[1, 1].legend()
        
        # Adjust spacing between subplots
        plt.tight_layout()
        plt.show()

        # Điều chỉnh khoảng cách giữa các subplots
        plt.tight_layout()
        plt.show()
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, 
                             QMessageBox, QVBoxLayout, QHBoxLayout, QWidget, QTextEdit, QDialog, QComboBox, QDialogButtonBox)
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

class CameraSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Camera to start")
        self.setGeometry(200, 200, 800, 150)
        
        layout = QVBoxLayout()
        
        # Camera selection dropdown
        self.camera_combo = QComboBox()
        layout.addWidget(QLabel("Available Cameras:"))
        layout.addWidget(self.camera_combo)
        
        # Populate camera list
        self.populate_camera_list()
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # If no cameras found, show error
        if self.camera_combo.count() == 0:
            QMessageBox.warning(self, "No Cameras", "No Basler cameras found.")
    
    def populate_camera_list(self):
        # Get list of available cameras
        camera_list = pylon.TlFactory.GetInstance().EnumerateDevices()
        
        for camera in camera_list:
            camera_info = f"{camera.GetSerialNumber()} - {camera.GetModelName()}"
            self.camera_combo.addItem(camera_info, userData=camera.GetSerialNumber())
    
    def get_selected_camera_serial(self):
        return self.camera_combo.currentData()

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
            self.camera.Width.Value = 1024  # Đặt chiều rộng hình ảnh
            self.camera.Height.Value = 600  # Đặt chiều cao hình ảnh

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
        prev_x, prev_y = None, None  # Lưu tọa độ trước đó của ROI

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

                        # Tính toán dx, dy (sự thay đổi tọa độ)
                        if prev_x is not None and prev_y is not None:
                            dx = found_x - prev_x
                            dy = found_y - prev_y
                            self.tracking_data[-1] += (dx, dy)  # Lưu thêm dx, dy vào dữ liệu
                        else:
                            dx = dy = 0

                        # Lưu dữ liệu tracking (bao gồm dx, dy)
                        self.tracking_data.append((found_x, found_y, confidence, dx, dy))

                        # Chuyển đổi sang không gian màu BGR nếu ảnh đang ở grayscale
                        if len(current_frame.shape) == 2:
                            current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2BGR)

                        cv2.rectangle(current_frame, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
                        self.display_image(current_frame)

                        resulting_fps = self.camera.ResultingFrameRate.Value
                        self.label_fps.setText(f"FPS: {resulting_fps:.2f}")

                    # Cập nhật số khung hình
                    self.frame_count += 1
                    self.frame_count_label.setText(f"Frames: {self.frame_count}")

                    # Cập nhật tọa độ trước đó
                    prev_x = found_x
                    prev_y = found_y

                    QApplication.processEvents()
                else:
                    QMessageBox.warning(self, "Error", "Capture Error: Failed to grab image")
                    self.stop_tracking()

            except Exception as e:
                QMessageBox.warning(self, "Error", f"Tracking Error: {str(e)}")
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





def main():
    app = QApplication(sys.argv)
    window = BaslerCameraROITracker()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

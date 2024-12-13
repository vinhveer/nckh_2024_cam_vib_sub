import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QPushButton, 
                             QMessageBox, QVBoxLayout, QHBoxLayout, QWidget, QDialog, 
                             QDialogButtonBox, QFormLayout, QLineEdit, QSlider, QSpinBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
from pypylon import pylon
from pyzbar import pyzbar
import matplotlib.pyplot as plt

from camera_selection_dialog import CameraSelectionDialog
from enhanced_roi_tracker import EnhancedROITracker

class BaslerCameraROITracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.tracking_data = []
        self.frame_count = 0
        self.setWindowTitle("Tracking ROI")
        
        # Hộp thoại chọn camera
        camera_dialog = CameraSelectionDialog(self)
        if camera_dialog.exec_() == QDialog.Accepted:
            selected_serial = camera_dialog.get_selected_camera_serial()
        else:
            QMessageBox.critical(self, "Error", "No camera selected. Exiting.")
            sys.exit(1)

        # Khởi tạo camera với số serial đã chọn
        try:
            tlFactory = pylon.TlFactory.GetInstance()
            devices = tlFactory.EnumerateDevices()
            
            # Tìm camera với số serial khớp
            selected_device = None
            for device in devices:
                if device.GetSerialNumber() == selected_serial:
                    selected_device = device
                    break
            
            if selected_device is None:
                raise RuntimeError("Selected camera not found")

            # Mở camera đã chọn
            self.camera = pylon.InstantCamera(tlFactory.CreateDevice(selected_device))
            self.camera.Open()

            # Đặt các tham số cố định cho camera
            self.camera.Width.Value = 2048
            self.camera.Height.Value = 2048

            # Lấy giới hạn thời gian phơi sáng
            self.min_exposure = self.camera.ExposureTime.GetMin()
            self.max_exposure = 49000  # Thời gian phơi sáng tối đa tùy chỉnh theo yêu cầu

            self.current_exposure = 10000  # Thời gian phơi sáng mặc định

            # Đặt thời gian phơi sáng ban đầu
            self.camera.ExposureTime.SetValue(self.current_exposure)

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
        self.current_frame = None

        # Bố cục
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Điều khiển thời gian phơi sáng
        exposure_layout = QHBoxLayout()
        self.exposure_label = QLabel(f"Exposure Time: {self.current_exposure}")
        exposure_layout.addWidget(self.exposure_label)

        # Tạo thanh trượt với phạm vi phơi sáng thực tế của camera
        self.exposure_slider = QSlider(Qt.Horizontal)
        self.exposure_slider.setMinimum(0)
        self.exposure_slider.setMaximum(1000)  # Chúng ta sẽ ánh xạ điều này tới phạm vi thực tế của camera
        self.exposure_slider.setValue(self.get_slider_value(self.current_exposure))
        self.exposure_slider.valueChanged.connect(self.update_exposure_time)
        exposure_layout.addWidget(self.exposure_slider)

        layout.addLayout(exposure_layout)

        # Nhãn đếm khung hình
        self.frame_count_label = QLabel("Frames: 0")
        layout.addWidget(self.frame_count_label)

        # Hiển thị FPS
        self.label_fps = QLabel("FPS: 0")
        layout.addWidget(self.label_fps)

        # Hiển thị kết quả theo dõi với chế độ xem trực tiếp
        self.label_tracked = QLabel("Live Camera View")
        self.label_tracked.setMinimumSize(800, 600)
        self.label_tracked.setAlignment(Qt.AlignCenter)
        self.label_tracked.setStyleSheet("border: 2px solid red;")
        layout.addWidget(self.label_tracked)

        # Bố cục nút
        button_layout = QHBoxLayout()
        layout.addLayout(button_layout)

        # Nút mới để chọn ROI trong chế độ xem trực tiếp
        self.select_roi_button = QPushButton("Select ROI")
        self.select_roi_button.clicked.connect(self.select_roi_during_live_view)
        button_layout.addWidget(self.select_roi_button)

        self.start_tracking_button = QPushButton("Start Tracking")
        self.start_tracking_button.clicked.connect(self.start_tracking)
        button_layout.addWidget(self.start_tracking_button)

        self.stop_tracking_button = QPushButton("Stop Tracking")
        self.stop_tracking_button.clicked.connect(self.stop_tracking)
        button_layout.addWidget(self.stop_tracking_button)

        # Bộ đếm thời gian để chụp khung hình liên tục
        self.frame_timer = QTimer(self)
        self.frame_timer.timeout.connect(self.capture_continuous_frame)
        self.frame_timer.start(33)  # Khoảng 30 FPS

        self.adjust_roi_button = QPushButton("Adjust Camera ROI")
        self.adjust_roi_button.clicked.connect(self.adjust_camera_roi)
        button_layout.addWidget(self.adjust_roi_button)

        # Store original full sensor dimensions
        self.full_width = 2048
        self.full_height = 2048
        self.current_roi = None  # Will store current ROI coordinates

    def adjust_camera_roi(self):
        """
        Chọn ROI bằng cách quét mã QR và theo dõi vị trí
        """
        # Dừng timer để tránh việc liên tục chụp khung hình
        self.frame_timer.stop()

        try:
            # Kiểm tra kết nối camera
            if not self.camera.IsOpen():
                QMessageBox.critical(self, "Lỗi", "Camera không được mở")
                return

            # Dừng grabbing nếu đang chạy
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()

            # Khởi tạo cờ để theo dõi việc chọn ROI
            roi_selected = False
            qr_location = None

            # Chuẩn bị cửa sổ để hiển thị
            cv2.namedWindow("QR Code Tracking", cv2.WINDOW_NORMAL)

            while not roi_selected:
                # Chụp khung hình
                try:
                    grab_result = self.camera.GrabOne(10000)
                    if grab_result is None or not grab_result.GrabSucceeded():
                        QMessageBox.critical(self, "Lỗi", "Không thể chụp khung hình")
                        self.frame_timer.start(33)
                        return

                    # Chuyển đổi khung hình
                    current_frame = grab_result.Array
                    
                    # Kiểm tra khung hình
                    if current_frame is None or current_frame.size == 0:
                        QMessageBox.critical(self, "Lỗi", "Không có dữ liệu khung hình")
                        self.frame_timer.start(33)
                        return

                except Exception as grab_error:
                    QMessageBox.critical(self, "Lỗi Chụp Khung Hình", f"Không thể chụp khung hình: {grab_error}")
                    self.frame_timer.start(33)
                    return

                # Chuyển đổi sang ảnh xám
                gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

                # Quét mã QR
                qr_codes = pyzbar.decode(gray)

                # Vẽ và theo dõi mã QR
                for qr in qr_codes:
                    # Lấy toạ độ
                    (x, y, w, h) = qr.rect
                    
                    # Vẽ hình chữ nhật xung quanh mã QR
                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Hiển thị thông tin mã QR
                    qr_data = qr.data.decode("utf-8")
                    cv2.putText(current_frame, qr_data, (x, y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Hiển thị khung hình
                cv2.imshow("QR Code Tracking", current_frame)

                # Chờ phím Enter để chọn ROI
                key = cv2.waitKey(1) & 0xFF
                if key == 13:  # Phím Enter
                    if qr_codes:
                        qr_location = qr_codes[0].rect
                        roi_selected = True
                    else:
                        QMessageBox.warning(self, "Cảnh Báo", "Không tìm thấy mã QR. Tiếp tục quét.")

                # Thoát nếu nhấn 'q'
                elif key == ord('q'):
                    self.frame_timer.start(33)
                    cv2.destroyWindow("QR Code Tracking")
                    return

            # Đóng cửa sổ tracking
            cv2.destroyWindow("QR Code Tracking")

            # Nếu không có mã QR
            if not qr_location:
                QMessageBox.warning(self, "Lỗi", "Không thể xác định ROI")
                self.frame_timer.start(33)
                return

            # Lấy các tham số căn chỉnh của camera
            try:
                # Lấy thông số căn chỉnh (increment) và giá trị Min/Max
                width_inc = getattr(self.camera.Width, 'Inc', 8)
                height_inc = getattr(self.camera.Height, 'Inc', 8)
                offset_x_inc = getattr(self.camera.OffsetX, 'Inc', 8)
                offset_y_inc = getattr(self.camera.OffsetY, 'Inc', 8)

                min_width = getattr(self.camera.Width, 'Min', 64)
                max_width = getattr(self.camera.Width, 'Max', 2048)
                min_height = getattr(self.camera.Height, 'Min', 64)
                max_height = getattr(self.camera.Height, 'Max', 2048)
                min_offset_x = getattr(self.camera.OffsetX, 'Min', 0)
                max_offset_x = getattr(self.camera.OffsetX, 'Max', 2048)
                min_offset_y = getattr(self.camera.OffsetY, 'Min', 0)
                max_offset_y = getattr(self.camera.OffsetY, 'Max', 2048)

            except Exception as attr_error:
                QMessageBox.warning(self, "Lỗi Thuộc Tính", f"Không thể truy xuất thuộc tính: {attr_error}")
                self.frame_timer.start(33)
                return

            # Điều chỉnh toạ độ ROI
            x, y, w, h = qr_location
            
            # Điều chỉnh để phù hợp với các ràng buộc của camera
            original_w = max(min_width, min(w, max_width))
            original_w = min_width + ((original_w - min_width) // width_inc * width_inc)

            original_h = max(min_height, min(h, max_height))
            original_h = min_height + ((original_h - min_height) // height_inc * height_inc)

            original_x = max(min_offset_x, min(x, max_offset_x - original_w))
            original_x = min_offset_x + ((original_x - min_offset_x) // offset_x_inc * offset_x_inc)

            original_y = max(min_offset_y, min(y, max_offset_y - original_h))
            original_y = min_offset_y + ((original_y - min_offset_y) // offset_y_inc * offset_y_inc)

            # Thiết lập ROI mới
            try:
                # Đặt từng giá trị một cách an toàn
                if hasattr(self.camera, 'OffsetX'):
                    self.camera.OffsetX.Value = original_x
                if hasattr(self.camera, 'OffsetY'):
                    self.camera.OffsetY.Value = original_y
                if hasattr(self.camera, 'Width'):
                    self.camera.Width.Value = original_w
                if hasattr(self.camera, 'Height'):
                    self.camera.Height.Value = original_h

                # Khởi động lại quá trình chụp
                self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

                # Khởi động lại timer
                self.frame_timer.start(33)

                # Thông báo ROI mới
                QMessageBox.information(self, "Cập Nhật ROI", 
                    f"ROI mới từ mã QR: X:{original_x}, Y:{original_y}, Rộng:{original_w}, Cao:{original_h}")

            except Exception as roi_error:
                QMessageBox.warning(self, "Lỗi Điều Chỉnh ROI", str(roi_error))
                self.frame_timer.start(33)

        except Exception as general_error:
            QMessageBox.critical(self, "Lỗi Chung", str(general_error))
            self.frame_timer.start(33)

    def reset_camera_roi(self):
        """
        Reset camera to full sensor dimensions
        """
        try:
            # Stop current acquisition
            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()

            # Reset to full sensor dimensions
            self.camera.OffsetX.Value = 0
            self.camera.OffsetY.Value = 0
            self.camera.Width.Value = self.full_width
            self.camera.Height.Value = self.full_height

            # Clear stored ROI
            self.current_roi = None

            # Restart grabbing
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

            QMessageBox.information(self, "ROI Reset", "Camera reset to full sensor dimensions.")

        except Exception as e:
            QMessageBox.critical(self, "ROI Reset Error", str(e))

    def get_slider_value(self, exposure_time):
        # Chuyển đổi thời gian phơi sáng thành giá trị thanh trượt (0-1000)
        # Ánh xạ tuyến tính từ khoảng thời gian phơi sáng sang khoảng giá trị thanh trượt
        return int(((exposure_time - self.min_exposure) / 
                    (self.max_exposure - self.min_exposure)) * 1000)

    def get_exposure_from_slider(self, slider_value):
        # Chuyển đổi giá trị thanh trượt thành thời gian phơi sáng thực tế
        # Ánh xạ tuyến tính từ khoảng giá trị thanh trượt sang khoảng thời gian phơi sáng
        return self.min_exposure + (slider_value / 1000) * (self.max_exposure - self.min_exposure)

    def update_exposure_time(self, slider_value):
        try:
            # Chuyển đổi giá trị thanh trượt thành thời gian phơi sáng thực tế
            exposure_time = self.get_exposure_from_slider(slider_value)
            
            # Thiết lập thời gian phơi sáng
            self.camera.ExposureTime.SetValue(exposure_time)
            
            # Cập nhật nhãn hiển thị
            self.exposure_label.setText(f"Exposure Time: {exposure_time:.2f}")
            
            # Lưu lại thời gian phơi sáng hiện tại
            self.current_exposure = exposure_time

        except Exception as e:
            QMessageBox.critical(self, "Lỗi thời gian phơi sáng", str(e))

    def select_roi_during_live_view(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "Lỗi", "Chưa có khung hình nào được chụp")
            return

        try:
            # Sử dụng hình ảnh gốc và giảm kích thước để hiển thị
            raw_frame = self.current_frame
            h, w = raw_frame.shape[:2]
            scaled_w, scaled_h = w // 2, h // 2
            
            # Thay đổi kích thước hình ảnh gốc
            scaled_image = cv2.resize(raw_frame, (scaled_w, scaled_h))

            # Hiển thị cửa sổ để chọn ROI
            cv2.imshow("Select ROI", scaled_image)

            # Chờ người dùng chọn ROI
            roi = cv2.selectROI("Select ROI", scaled_image, showCrosshair=True, fromCenter=False)

            # Đóng cửa sổ hiển thị
            cv2.destroyWindow("Select ROI")

            # Chuyển đổi toạ độ ROI về kích thước gốc
            x, y, w, h = roi
            original_x = int(x * 2)
            original_y = int(y * 2)
            original_w = int(w * 2)
            original_h = int(h * 2)

            if original_w > 0 and original_h > 0:
                # Cắt vùng template từ hình ảnh gốc
                self.template = self.current_frame[original_y:original_y+original_h, original_x:original_x+original_w]
                self.roi_rect = (original_x, original_y, original_x + original_w, original_y + original_h)
                QMessageBox.information(self, "ROI đã chọn", f"ROI: {self.roi_rect}")
            else:
                QMessageBox.warning(self, "Lỗi", "Chọn ROI không hợp lệ")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi", str(e))

    def capture_continuous_frame(self):
        try:
            grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grabResult.GrabSucceeded():
                self.current_frame = grabResult.Array
                
                # Hiển thị khung hình
                if len(self.current_frame.shape) == 2:
                    self.current_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_GRAY2BGR)
                
                self.display_image(self.current_frame)

                # Cập nhật FPS
                resulting_fps = round(self.camera.ResultingFrameRate.Value, 8)
                self.label_fps.setText(f"FPS: {resulting_fps:.8f}")
        except Exception as e:
            QMessageBox.critical(self, "Lỗi chụp khung hình", str(e))

    def display_image(self, image):
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.label_tracked.setPixmap(pixmap.scaled(self.label_tracked.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_tracking(self):
        # Initial condition check
        if self.template is None:
            QMessageBox.warning(self, "Error", "Please select ROI first")
            return

        # Stop continuous frame capture
        self.frame_timer.stop()
        
        # Tracking mode
        self.is_tracking = True
        self.track()

    def track(self):
        prev_x, prev_y = None, None

        while self.is_tracking:
            try:
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    current_frame = grabResult.Array
                    x1, y1, x2, y2 = self.roi_rect
                    center_x = round(float(x1 + x2) / 2, 8)
                    center_y = round(float(y1 + y2) / 2, 8)

                    result = EnhancedROITracker.template_match(current_frame, self.template, center_x, center_y, initial_search_area=100)

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
        
        # Restart continuous frame capture
        self.frame_timer.start(33)
        
        self.show_plot()

    def show_plot(self):
        if not self.tracking_data:
            QMessageBox.information(self, "No Data", "No tracking data to display.")
            return
        
        # Extract data
        dx_data = [data[3] for data in self.tracking_data]
        dy_data = [data[4] for data in self.tracking_data]
        
        # Sampling frequency (assumed)
        Fs = 240  # Hz
        N = len(dx_data)  # Number of data points
        t = np.linspace(0, N/Fs, N)  # Create time vector
        
        # Create figure with 2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))
        
        # dx plot (left)
        axs[0].plot(dx_data, label="dx", color='blue')
        axs[0].set_title("dx vs Time")
        axs[0].set_xlabel("Frame")
        axs[0].set_ylabel("dx")
        axs[0].grid()
        axs[0].legend()
        
        # dy plot (right)
        axs[1].plot(dy_data, label="dy", color='red')
        axs[1].set_title("dy vs Time")
        axs[1].set_xlabel("Frame")
        axs[1].set_ylabel("dy")
        axs[1].grid()
        axs[1].legend()
        
        # Adjust spacing between subplots
        plt.tight_layout()
        plt.show()

    # def show_plot(self):
    #     if not self.tracking_data:
    #         QMessageBox.information(self, "No Data", "No tracking data to display.")
    #         return
        
    #     # Chuẩn bị dữ liệu cho từng phương pháp
    #     methods = ['Gaussian', 'Quadratic', 'Interpolation']
        
    #     # Tạo figure 6 subplots
    #     fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    #     fig.suptitle('Sub-Pixel Refinement Methods Comparison', fontsize=16)
        
    #     # Lặp qua từng phương pháp và vẽ đồ thị dx, dy
    #     for i, method in enumerate(methods):
    #         # Lấy dữ liệu dx và dy cho phương pháp hiện tại
    #         dx_data = [data[3][method][0] for data in self.tracking_data]
    #         dy_data = [data[3][method][1] for data in self.tracking_data]
            
    #         # dx plot
    #         axs[0, i].plot(dx_data, label=f"{method} dx", color='blue')
    #         axs[0, i].set_title(f"{method} dx vs Time")
    #         axs[0, i].set_xlabel("Frame")
    #         axs[0, i].set_ylabel("dx")
    #         axs[0, i].grid()
    #         axs[0, i].legend()
            
    #         # dy plot
    #         axs[1, i].plot(dy_data, label=f"{method} dy", color='red')
    #         axs[1, i].set_title(f"{method} dy vs Time")
    #         axs[1, i].set_xlabel("Frame")
    #         axs[1, i].set_ylabel("dy")
    #         axs[1, i].grid()
    #         axs[1, i].legend()
        
    #     # Điều chỉnh khoảng cách giữa các subplot
    #     plt.tight_layout()
    #     plt.show()
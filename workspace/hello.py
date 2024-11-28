import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QFileDialog, 
                             QPushButton, QMessageBox, QVBoxLayout, QHBoxLayout, 
                             QWidget, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class EnhancedROITracker:
    @staticmethod
    def multi_method_tracking(image, template, initial_x, initial_y, initial_search_area=50):
        # Chuyển đổi sang ảnh xám
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        
        # Phương pháp 1: Template Matching
        template_result = EnhancedROITracker._template_match(gray_image, gray_template, initial_x, initial_y, initial_search_area)
        
        # Phương pháp 2: Feature Matching (ORB)
        feature_result = EnhancedROITracker._feature_match(gray_image, gray_template, initial_x, initial_y, initial_search_area)
        
        # Kết hợp kết quả từ các phương pháp
        combined_results = [r for r in [template_result, feature_result] if r is not None]
        
        if not combined_results:
            return None
        
        # Chọn kết quả có độ tin cậy cao nhất
        best_result = max(combined_results, key=lambda x: x[2])
        return best_result

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
            
            # Sử dụng phương pháp matching tốt nhất
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
            
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.65:
                found_x = x1 + max_loc[0] + w//2
                found_y = y1 + max_loc[1] + h//2
                return found_x, found_y, max_val
            
            search_area *= 2
        
        return None

    @staticmethod
    def _feature_match(image, template, initial_x, initial_y, initial_search_area):
        # Sử dụng ORB detector và descriptor
        orb = cv2.ORB_create()
        
        # Tìm keypoints và descriptors
        kp1, des1 = orb.detectAndCompute(template, None)
        kp2, des2 = orb.detectAndCompute(image, None)
        
        # Matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        
        # Sắp xếp matches theo khoảng cách
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Lọc matches gần điểm ban đầu
        good_matches = [
            m for m in matches 
            if (abs(kp2[m.trainIdx].pt[0] - initial_x) <= initial_search_area and
                abs(kp2[m.trainIdx].pt[1] - initial_y) <= initial_search_area)
        ]
        
        if good_matches:
            # Chọn match tốt nhất
            best_match = good_matches[0]
            found_x, found_y = kp2[best_match.trainIdx].pt
            confidence = 1 - (best_match.distance / 500)  # Normalize confidence
            
            return (int(found_x), int(found_y), confidence)
        
        return None

class ImageROITracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Enhanced ROI Neighborhood Tracking")
        self.setGeometry(100, 100, 1400, 800)

        # Tạo widget trung tâm và layout chính
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Layout hình ảnh
        image_layout = QHBoxLayout()
        
        # Label hiển thị ảnh gốc
        self.label_original = QLabel("Image 1")
        self.label_original.setMinimumSize(600, 600)
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("border: 2px solid blue;")
        self.label_original.mousePressEvent = self.start_roi_selection
        self.label_original.mouseReleaseEvent = self.end_roi_selection
        self.label_original.mouseMoveEvent = self.update_roi_selection

        # Label hiển thị ảnh sau tracking
        self.label_tracked = QLabel("Tracked Image")
        self.label_tracked.setMinimumSize(600, 600)
        self.label_tracked.setAlignment(Qt.AlignCenter)
        self.label_tracked.setStyleSheet("border: 2px solid red;")

        # Thêm hai label vào layout
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_tracked)
        main_layout.addLayout(image_layout)

        # Khung hiển thị thông tin
        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)
        main_layout.addWidget(self.text_info)

        # Layout nút bấm
        button_layout = QHBoxLayout()
        
        # Các nút chức năng
        self.load_image1_button = QPushButton("Load Image 1")
        self.load_image2_button = QPushButton("Load Image 2")
        self.track_button = QPushButton("Track ROI")
        self.track_button.setEnabled(False)

        # Kết nối sự kiện cho các nút
        self.load_image1_button.clicked.connect(self.load_image1)
        self.load_image2_button.clicked.connect(self.load_image2)
        self.track_button.clicked.connect(self.track_roi)

        # Thêm nút vào layout
        button_layout.addWidget(self.load_image1_button)
        button_layout.addWidget(self.load_image2_button)
        button_layout.addWidget(self.track_button)
        main_layout.addLayout(button_layout)

        # Các biến lưu trữ
        self.image1 = None
        self.image2 = None
        self.roi_start = None
        self.roi_end = None
        self.roi_rect = None
        self.is_selecting = False
        self.roi_template = None

    def load_image1(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image 1", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image1 = cv2.imread(file_name)
            self.display_image(self.label_original, self.image1)

    def load_image2(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image 2", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_name:
            self.image2 = cv2.imread(file_name)
            self.display_image(self.label_tracked, self.image2)
            self.track_button.setEnabled(True)

    def display_image(self, label, image):
        if image is not None:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def start_roi_selection(self, event):
        if self.image1 is not None:
            self.roi_start = event.pos()
            self.is_selecting = True

    def update_roi_selection(self, event):
        if self.is_selecting and self.image1 is not None:
            self.roi_end = event.pos()
            image_copy = self.image1.copy()
            x1, y1, x2, y2 = self.get_roi_coordinates()
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            self.display_image(self.label_original, image_copy)

    def end_roi_selection(self, event):
        if self.is_selecting:
            self.is_selecting = False
            self.roi_end = event.pos()
            x1, y1, x2, y2 = self.get_roi_coordinates()
            
            if x1 < x2 and y1 < y2:
                self.roi_rect = (x1, y1, x2, y2)
                # Tạo template ROI
                self.roi_template = self.image1[y1:y2, x1:x2]
                self.text_info.append(f"ROI Selected: ({x1}, {y1}) to ({x2}, {y2})")
            else:
                QMessageBox.warning(self, "Error", "Invalid ROI selection")

    def get_roi_coordinates(self):
        if self.roi_start and self.roi_end and self.image1 is not None:
            label_width = self.label_original.width()
            label_height = self.label_original.height()
            
            x_ratio = self.image1.shape[1] / self.label_original.pixmap().width()
            y_ratio = self.image1.shape[0] / self.label_original.pixmap().height()

            x1 = int(self.roi_start.x() * x_ratio)
            y1 = int(self.roi_start.y() * y_ratio)
            x2 = int(self.roi_end.x() * x_ratio)
            y2 = int(self.roi_end.y() * y_ratio)

            return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        return 0, 0, 0, 0

    def track_roi(self):
        if (self.image1 is not None and 
            self.image2 is not None and 
            self.roi_rect is not None and
            self.roi_template is not None):
            
            x1, y1, x2, y2 = self.roi_rect
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # Chuyển sang ảnh xám
            gray_image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
            
            # Thực hiện tìm kiếm lân cận
            result = EnhancedROITracker.multi_method_tracking(
                self.image2, 
                self.roi_template, 
                center_x, 
                center_y, 
                initial_search_area=100
            )
            
            if result:
                found_x, found_y, confidence = result
                
                # Tính toán kích thước ROI
                w = x2 - x1
                h = y2 - y1
                
                # Điều chỉnh tọa độ ROI mới
                new_x1 = found_x - w//2
                new_y1 = found_y - h//2
                new_x2 = new_x1 + w
                new_y2 = new_y1 + h
                
                # Vẽ khung ROI mới
                tracked_image = self.image2.copy()
                cv2.rectangle(tracked_image, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)
                
                # Hiển thị ảnh và thông tin
                self.display_image(self.label_tracked, tracked_image)
                
                # Ghi thông tin di chuyển
                dx = found_x - center_x
                dy = found_y - center_y
                self.text_info.append(f"ROI Moved: dx = {dx}, dy = {dy}")
                self.text_info.append(f"Confidence: {confidence:.2f}")
                self.text_info.append(f"Old Center: ({center_x}, {center_y})")
                self.text_info.append(f"New Center: ({found_x}, {found_y})")
            else:
                QMessageBox.warning(self, "Error", "ROI tracking failed")

# Chạy ứng dụng
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ImageROITracker()
    main_window.show()
    sys.exit(app.exec_())
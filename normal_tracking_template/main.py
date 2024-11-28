# main.py
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QLabel, QPushButton, QListWidget, QWidget, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
from camera_module import CameraManager
from template_selector import TemplateSelector
from template_tracker import TemplateTracker
import cv2

class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Template Tracking Camera")
        self.setGeometry(100, 100, 800, 600)

        # Modules
        self.camera_manager = CameraManager()
        self.template_selector = TemplateSelector()
        self.template_tracker = TemplateTracker()

        # UI components
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)

        self.camera_list_widget = QListWidget()
        self.refresh_button = QPushButton("Làm mới danh sách Camera")
        self.start_button = QPushButton("Bắt đầu chụp")
        self.track_button = QPushButton("Tracking Template")
        self.image_label = QLabel("Hình ảnh Camera")
        self.image_label.setFixedSize(720, 480)
        self.image_label.setStyleSheet("border: 2px solid gray;")

        control_layout = QHBoxLayout()
        control_layout.addWidget(self.refresh_button)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.track_button)

        self.layout.addWidget(self.camera_list_widget)
        self.layout.addLayout(control_layout)
        self.layout.addWidget(self.image_label)

        # Event connections
        self.refresh_button.clicked.connect(self.refresh_camera_list)
        self.start_button.clicked.connect(self.start_camera)
        self.track_button.clicked.connect(self.track_template)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.refresh_camera_list()

    def refresh_camera_list(self):
        cameras = self.camera_manager.refresh_camera_list()
        self.camera_list_widget.clear()
        for name, serial in cameras:
            self.camera_list_widget.addItem(f"{name} (SN: {serial})")

    def start_camera(self):
        index = self.camera_list_widget.currentRow()
        self.camera_manager.select_camera(index)
        self.timer.start(50)

    def update_frame(self):
        frame = self.camera_manager.get_frame()
        if frame is not None:
            if self.template_tracker.template is not None:
                _, bbox = self.template_tracker.track(frame)
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            height, width, channel = frame.shape
            bytes_per_line = channel * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image).scaled(self.image_label.size())
            self.image_label.setPixmap(pixmap)

    def track_template(self):
        frame = self.camera_manager.get_frame()
        if frame is not None:
            # Sử dụng TemplateSelector (OpenCV) để chọn template
            template, bbox = self.template_selector.set_template(frame)
            if template is not None and bbox is not None:
                self.template_tracker.set_template(template, bbox)
            else:
                print("Template selection failed or was canceled.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CameraApp()
    window.show()
    sys.exit(app.exec_())

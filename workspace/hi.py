import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QLabel, QMainWindow, QFileDialog, 
                             QPushButton, QMessageBox, QVBoxLayout, QHBoxLayout, 
                             QWidget, QTextEdit)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class EnhancedFeatureTracker:
    @staticmethod
    def feature_tracking(image, template, initial_x, initial_y, initial_search_area=50):
        detector_types = [
            ('orb', cv2.ORB_create(nfeatures=1000)),
            ('sift', cv2.SIFT_create()),
            ('kaze', cv2.KAZE_create())
        ]
        
        best_match_result = None
        best_confidence = 0
        
        for detector_name, detector in detector_types:
            try:
                kp1, des1 = detector.detectAndCompute(template, None)
                kp2, des2 = detector.detectAndCompute(image, None)

                print(f"Detector: {detector_name} - Keypoints in template: {len(kp1)}, Keypoints in image: {len(kp2)}")

                if len(kp1) == 0 or len(kp2) == 0 or des1 is None or des2 is None:
                    print(f"No keypoints or descriptors found for {detector_name}")
                    continue
                
                # Choose the appropriate distance metric
                if detector_name == 'orb':
                    norm_type = cv2.NORM_HAMMING
                else:
                    norm_type = cv2.NORM_L2

                matcher = cv2.BFMatcher(norm_type, crossCheck=True)
                matches = matcher.match(des1, des2)
                matches = sorted(matches, key=lambda x: x.distance)

                # Filter matches based on initial search area
                good_matches = [
                    m for m in matches 
                    if (abs(kp2[m.trainIdx].pt[0] - initial_x) <= initial_search_area and
                        abs(kp2[m.trainIdx].pt[1] - initial_y) <= initial_search_area)
                ]

                if good_matches:
                    best_match = good_matches[0]
                    found_x, found_y = kp2[best_match.trainIdx].pt
                    confidence = 1 - (best_match.distance / 500)

                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match_result = (int(found_x), int(found_y), confidence)
            
            except Exception as e:
                print(f"Error with {detector_name} detector: {e}")
        
        return best_match_result

class ImageROITracker(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Feature-Based ROI Tracking")
        self.setGeometry(100, 100, 1400, 800)

        # Create main widget and layout
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Layout for images
        image_layout = QHBoxLayout()

        # Label for displaying the original image
        self.label_original = QLabel("Image 1")
        self.label_original.setMinimumSize(600, 600)
        self.label_original.setAlignment(Qt.AlignCenter)
        self.label_original.setStyleSheet("border: 2px solid blue;")
        self.label_original.mousePressEvent = self.start_roi_selection
        self.label_original.mouseReleaseEvent = self.end_roi_selection
        self.label_original.mouseMoveEvent = self.update_roi_selection

        # Label for displaying the tracked image
        self.label_tracked = QLabel("Tracked Image")
        self.label_tracked.setMinimumSize(600, 600)
        self.label_tracked.setAlignment(Qt.AlignCenter)
        self.label_tracked.setStyleSheet("border: 2px solid red;")

        # Add the labels to the layout
        image_layout.addWidget(self.label_original)
        image_layout.addWidget(self.label_tracked)
        main_layout.addLayout(image_layout)

        # Text box for displaying information
        self.text_info = QTextEdit()
        self.text_info.setReadOnly(True)
        main_layout.addWidget(self.text_info)

        # Button layout
        button_layout = QHBoxLayout()
        self.load_image1_button = QPushButton("Load Image 1")
        self.load_image2_button = QPushButton("Load Image 2")
        self.track_button = QPushButton("Track ROI")
        self.track_button.setEnabled(False)

        # Connect button events
        self.load_image1_button.clicked.connect(self.load_image1)
        self.load_image2_button.clicked.connect(self.load_image2)
        self.track_button.clicked.connect(self.track_roi)

        # Add buttons to the layout
        button_layout.addWidget(self.load_image1_button)
        button_layout.addWidget(self.load_image2_button)
        button_layout.addWidget(self.track_button)
        main_layout.addLayout(button_layout)

        # Variables to store images and ROI information
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
                self.roi_template = self.image1[y1:y2, x1:x2]
                self.text_info.append(f"ROI Selected: ({x1}, {y1}) to ({x2}, {y2})")
            else:
                QMessageBox.warning(self, "Error", "Invalid ROI selection")

    def get_roi_coordinates(self):
        if self.roi_start and self.roi_end and self.image1 is not None:
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

            gray_image2 = cv2.cvtColor(self.image2, cv2.COLOR_BGR2GRAY)
            result = EnhancedFeatureTracker.feature_tracking(
                gray_image2, 
                self.roi_template, 
                center_x, 
                center_y, 
                initial_search_area=100
            )

            if result:
                h = y2 - y1
                w = x2 - x1
                new_x, new_y, confidence = result
                self.text_info.append(f"Tracking Result: ({new_x}, {new_y}), Confidence: {confidence:.2f}")

                image_with_tracking = self.image2.copy()
                cv2.rectangle(image_with_tracking, (new_x - w//2, new_y - h//2), 
                              (new_x + w//2, new_y + h//2), (0, 255, 0), 2)
                self.display_image(self.label_tracked, image_with_tracking)
            else:
                QMessageBox.warning(self, "Error", "Failed to track ROI")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageROITracker()
    window.show()
    sys.exit(app.exec_())

import cv2
import numpy as np

class TemplateTracker:
    def __init__(self):
        self.template = None
        self.bbox = None
        self.threshold = 0.8  # Ngưỡng tương quan để chấp nhận kết quả khớp

    def set_template(self, template, bbox):
        """Thiết lập template và vị trí ban đầu"""
        self.template = template
        self.bbox = bbox

    def track(self, frame):
        """Theo dõi template trên khung hình"""
        if self.template is None:
            print("Template chưa được thiết lập!")
            return None, None

        # Sử dụng cv2.matchTemplate để tìm template
        result = cv2.matchTemplate(frame, self.template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # Kiểm tra giá trị tương quan có vượt ngưỡng không
        if max_val >= self.threshold:
            top_left = max_loc
            w, h = self.template.shape[1], self.template.shape[0]
            self.bbox = (top_left[0], top_left[1], w, h)
            return frame, self.bbox
        else:
            print(f"Không tìm thấy template với độ khớp cao hơn ngưỡng {self.threshold}")
            return frame, None

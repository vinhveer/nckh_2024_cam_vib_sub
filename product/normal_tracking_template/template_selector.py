import cv2

class TemplateSelector:
    def __init__(self):
        self.template = None
        self.bbox = None

    def set_template(self, frame):
        """Thiết lập template từ khung hình bằng cách sử dụng cv2.selectROI"""
        # Hiển thị cửa sổ cho phép người dùng chọn vùng
        roi = cv2.selectROI("Select Template", frame, fromCenter=False, showCrosshair=True)
        x, y, w, h = roi

        # Kiểm tra nếu ROI hợp lệ
        if w > 0 and h > 0:
            self.template = frame[y:y+h, x:x+w]
            self.bbox = (x, y, w, h)
        else:
            print("Invalid ROI selected")

        # Đóng cửa sổ
        cv2.destroyWindow("Select Template")

        return self.template, self.bbox

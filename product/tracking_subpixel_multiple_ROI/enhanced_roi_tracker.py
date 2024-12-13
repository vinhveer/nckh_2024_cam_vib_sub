import cv2
import numpy as np

class EnhancedROITracker:
    @staticmethod
    def template_match(image, template, initial_x, initial_y, initial_search_area=50):
        # Chuyển đổi ảnh và template sang grayscale nếu cần
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template
        h, w = gray_template.shape[:2]
        search_area = initial_search_area
        while search_area <= max(gray_image.shape[:2]):
            # Xác định vùng tìm kiếm
            x1 = max(0, initial_x - search_area)
            y1 = max(0, initial_y - search_area)
            x2 = min(gray_image.shape[1], initial_x + search_area)
            y2 = min(gray_image.shape[0], initial_y + search_area)
            search_region = gray_image[int(y1):int(y2), int(x1):int(x2)]
            result = cv2.matchTemplate(search_region, gray_template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            if max_val > 0.65:
                # Vị trí tương đối trong search region
                rel_x, rel_y = max_loc[0], max_loc[1]
                # Tính tọa độ toàn cục
                found_x = x1 + rel_x + w / 2.0
                found_y = y1 + rel_y + h / 2.0
                # Nội suy sub-pixel bằng Quadratic fitting
                sub_pixel_offset_x, sub_pixel_offset_y = EnhancedROITracker._subpixel_refinement_quadratic(result, max_loc)
                found_x = round(found_x + sub_pixel_offset_x, 8)
                found_y = round(found_y + sub_pixel_offset_y, 8)
                return found_x, found_y, round(max_val, 8)
            search_area *= 2
        return None
   
    @staticmethod
    def _subpixel_refinement_quadratic(result, max_loc):
        x, y = max_loc
        h, w = result.shape
        
        # Ma trận T đã cho
        T = np.array([[1/6, -1/3, 1/6, 1/6, -1/3, 1/6, 1/6, -1/3, 1/6],
                      [-1/4, 0, 1/4, 0, 0, 0, 1/4, 0, -1/4],
                      [1/6, 1/6, 1/6, -1/3, -1/3, -1/3, 1/6, 1/6, 1/6],
                      [-1/6, 0, 1/6, -1/6, 0, 1/6, -1/6, 0, 1/6],
                      [1/6, 1/6, 1/6, 0, 0, 0, -1/6, -1/6, -1/6],
                      [-1/9, 2/9, -1/9, 2/9, 5/9, 2/9, -1/9, 2/9, -1/9]])
        
        # Lấy các giá trị xung quanh điểm max_loc trong kết quả
        S = np.array([result[y-1, x-1], result[y-1, x], result[y-1, x+1],
                      result[y, x-1], result[y, x], result[y, x+1],
                      result[y+1, x-1], result[y+1, x], result[y+1, x+1]])
        
        # Tính toán r = T * S
        r = np.dot(T, S)
        
        # Các giá trị a, b, c, d, e, f từ kết quả r
        a, b, c, d, e, f = r[:6]
        
        # Tính tọa độ sub-pixel
        xs = (2*c*d - b*e) / (b**2 - 4*a*c)
        ys = (2*a*e - b*d) / (b**2 - 4*a*c)

        return xs, ys
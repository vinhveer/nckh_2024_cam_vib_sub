# import cv2
# import numpy as np

# class EnhancedROITracker:
#     @staticmethod
#     def template_match(image, template, initial_x, initial_y, initial_search_area=50):
#         # Chuyển đổi ảnh và template sang grayscale nếu cần
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
#         gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

#         h, w = gray_template.shape[:2]
#         search_area = initial_search_area

#         while search_area <= max(gray_image.shape[:2]):
#             # Xác định vùng tìm kiếm
#             x1 = max(0, initial_x - search_area)
#             y1 = max(0, initial_y - search_area)
#             x2 = min(gray_image.shape[1], initial_x + search_area)
#             y2 = min(gray_image.shape[0], initial_y + search_area)

#             search_region = gray_image[int(y1):int(y2), int(x1):int(x2)]
#             result = cv2.matchTemplate(search_region, gray_template, cv2.TM_CCOEFF_NORMED)

#             _, max_val, _, max_loc = cv2.minMaxLoc(result)

#             if max_val > 0.65:
#                 # Vị trí tương đối trong search region
#                 rel_x, rel_y = max_loc[0], max_loc[1]

#                 # Tính tọa độ toàn cục
#                 found_x = x1 + rel_x + w / 2.0
#                 found_y = y1 + rel_y + h / 2.0

#                 # Nội suy sub-pixel bằng Gaussian fitting
#                 sub_pixel_offset_x, sub_pixel_offset_y = EnhancedROITracker._subpixel_refinement_gaussian(result, max_loc)
#                 found_x = round(found_x + sub_pixel_offset_x, 8)
#                 found_y = round(found_y + sub_pixel_offset_y, 8)

#                 return found_x, found_y, round(max_val, 8)

#             search_area *= 2

#         return None
    
#     @staticmethod
#     def _subpixel_refinement_gaussian(result, max_loc):
#         x, y = max_loc
#         h, w = result.shape

#         # Gaussian Fitting
#         try:
#             dx = [
#                 result[y, x - 1] if x > 0 else 0,
#                 result[y, x],
#                 result[y, x + 1] if x < w - 1 else 0
#             ]
#             dy = [
#                 result[y - 1, x] if y > 0 else 0,
#                 result[y, x],
#                 result[y + 1, x] if y < h - 1 else 0
#             ]

#             offset_x_gaussian = (dx[2] - dx[0]) / (2 * (2 * dx[1] - dx[0] - dx[2]) + 1e-8)
#             offset_y_gaussian = (dy[2] - dy[0]) / (2 * (2 * dy[1] - dy[0] - dy[2]) + 1e-8)

#             return offset_x_gaussian, offset_y_gaussian
#         except Exception:
#             return 0, 0

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMessageBox

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

                # Thực hiện 3 phương pháp nội suy sub-pixel
                sub_pixel_methods = {
                    'Gaussian': EnhancedROITracker._subpixel_refinement_gaussian(result, max_loc),
                    'Quadratic': EnhancedROITracker._subpixel_refinement_quadratic(result, max_loc),
                    'Interpolation': EnhancedROITracker._subpixel_refinement_interpolation(result, max_loc)
                }

                # Tạo kết quả cuối cùng
                final_results = {}
                for method, (offset_x, offset_y) in sub_pixel_methods.items():
                    refined_x = round(found_x + offset_x, 8)
                    refined_y = round(found_y + offset_y, 8)
                    final_results[method] = (refined_x, refined_y)

                return final_results, round(max_val, 8)

            search_area *= 2

        return None
    
    @staticmethod
    def _subpixel_refinement_gaussian(result, max_loc):
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

            return offset_x_gaussian, offset_y_gaussian
        except Exception:
            return 0, 0

    @staticmethod
    def _subpixel_refinement_quadratic(result, max_loc):
        x, y = max_loc
        h, w = result.shape

        try:
            # Quadratic fitting theo mẫu thuật toán
            T = np.array([
                [1/6, -1/3, 1/6, 1/6, -1/3, 1/6, 1/6, -1/3, 1/6],
                [-1/4, 0, 1/4, 0, 0, 0, 1/4, 0, -1/4],
                [1/6, 1/6, 1/6, -1/3, -1/3, -1/3, 1/6, 1/6, 1/6],
                [-1/6, 0, 1/6, -1/6, 0, 1/6, -1/6, 0, 1/6],
                [1/6, 1/6, 1/6, 0, 0, 0, -1/6, -1/6, -1/6],
                [-1/9, 2/9, -1/9, 2/9, 5/9, 2/9, -1/9, 2/9, -1/9]
            ])

            # Lấy các giá trị xung quanh điểm cực đại
            S = np.array([
                result[y-1, x-1], result[y-1, x], result[y-1, x+1],
                result[y, x-1], result[y, x], result[y, x+1],
                result[y+1, x-1], result[y+1, x], result[y+1, x+1]
            ])

            r = T @ S
            a, b, c = r[0], r[1], r[2]
            d, e, f = r[3], r[4], r[5]

            # Tính offset sub-pixel
            xs = (2*c*d - b*e) / (b**2 - 4*a*c + 1e-8)
            ys = (2*a*e - b*d) / (b**2 - 4*a*c + 1e-8)

            return xs, ys
        except Exception:
            return 0, 0

    @staticmethod
    def _subpixel_refinement_interpolation(result, max_loc):
        x, y = max_loc
        h, w = result.shape

        try:
            # Resize khu vực quanh điểm cực đại để nội suy
            region_size = 3
            region = result[max(0, y-1):min(h, y+2), max(0, x-1):min(w, x+2)]
            
            # Resize 2 lần để nội suy
            high_res_region = cv2.resize(region, (region_size*2, region_size*2), interpolation=cv2.INTER_CUBIC)
            
            # Tìm điểm cực đại mới trong vùng nội suy
            _, _, _, max_loc_interp = cv2.minMaxLoc(high_res_region)
            
            # Tính offset
            offset_x = (max_loc_interp[0] - region_size) / 2
            offset_y = (max_loc_interp[1] - region_size) / 2

            return offset_x, offset_y
        except Exception:
            return 0, 0
import cv2
import numpy as np
from PIL import Image

class EnhancedROITracker:
    @staticmethod
    def template_match(image, template, initial_x, initial_y, initial_search_area=50, mode = 'quadratic'):
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
                if mode == 'quadratic':
                    sub_pixel_offset_x, sub_pixel_offset_y = EnhancedROITracker._subpixel_refinement_quadratic(result, max_loc)
                elif mode == 'gaussian':
                    sub_pixel_offset_x, sub_pixel_offset_y = EnhancedROITracker._subpixel_refinement_gaussian(result, max_loc)
                elif mode == 'increase':
                    sub_pixel_offset_x, sub_pixel_offset_y = EnhancedROITracker._subpixel_refinement_increase(result, max_loc)

                found_x = round(found_x + sub_pixel_offset_x, 8)
                found_y = round(found_y + sub_pixel_offset_y, 8)
                return found_x, found_y, round(max_val, 8)
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
    
    @staticmethod
    def _subpixel_refinement_increase(result, max_loc):
        """
        Increase the resolution of a given image region around max_loc using interpolation.

        Args:
            result (np.ndarray): The input 2D array representing the image or region of interest.
            max_loc (tuple): The (x, y) location of the point to refine.

        Returns:
            tuple: The refined sub-pixel (x, y) location.
        """
        x, y = max_loc
        
        h, w = result.shape

        # Scale factor for refinement (example: double resolution)
        scale_factor = 2

        # Create a new higher-resolution result array
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        refined_result = np.zeros((new_h, new_w))

        for i in range(new_h):
            for j in range(new_w):
                # Map back to original space
                orig_y = i / scale_factor
                orig_x = j / scale_factor

                y0 = int(orig_y)
                x0 = int(orig_x)
                y1 = min(y0 + 1, h - 1)
                x1 = min(x0 + 1, w - 1)

                # Bilinear interpolation
                dy = orig_y - y0
                dx = orig_x - x0

                refined_result[i, j] = (
                    result[y0, x0] * (1 - dx) * (1 - dy) +
                    result[y0, x1] * dx * (1 - dy) +
                    result[y1, x0] * (1 - dx) * dy +
                    result[y1, x1] * dx * dy
                )

        # Find the new maximum location
        new_max_loc = np.unravel_index(np.argmax(refined_result), refined_result.shape)
        new_x, new_y = new_max_loc[1], new_max_loc[0]

        # Refine the coordinates back to original space
        refined_x = new_x / scale_factor
        refined_y = new_y / scale_factor

        return refined_x, refined_y

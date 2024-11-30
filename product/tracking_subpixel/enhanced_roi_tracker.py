import cv2
import numpy as np

class EnhancedROITracker:
    @staticmethod
    def multi_method_tracking(image, template, initial_x, initial_y, initial_search_area=50, prev_image=None, method="auto"):
        # Chuyển đổi ảnh và template sang grayscale nếu cần
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if len(template.shape) == 3 else template

        # Xử lý dựa trên phương pháp được chọn
        if method == "feature_match":
            return EnhancedROITracker._feature_match(gray_image, gray_template, initial_x, initial_y, initial_search_area)
        elif method == "optical_flow" and prev_image is not None:
            return EnhancedROITracker._optical_flow(prev_image, gray_image, initial_x, initial_y)
        elif method == "template_match":
            return EnhancedROITracker._template_match(gray_image, gray_template, initial_x, initial_y, initial_search_area)
        elif method == "auto":
            # Thử feature matching trước
            feature_result = EnhancedROITracker._feature_match(gray_image, gray_template, initial_x, initial_y, initial_search_area)

            # Nếu feature matching thất bại, thử optical flow
            if feature_result is None and prev_image is not None:
                optical_flow_result = EnhancedROITracker._optical_flow(prev_image, gray_image, initial_x, initial_y)
                if optical_flow_result is not None:
                    return optical_flow_result

            # Nếu cả hai phương pháp trên thất bại, chuyển sang template matching
            if feature_result is None:
                template_result = EnhancedROITracker._template_match(gray_image, gray_template, initial_x, initial_y, initial_search_area)
                return template_result

            return feature_result
        else:
            raise ValueError(f"Unsupported method: {method}")

    @staticmethod
    def _template_match(image, template, initial_x, initial_y, initial_search_area):
        h, w = template.shape[:2]
        search_area = initial_search_area

        while search_area <= max(image.shape[:2]):
            # Xác định vùng tìm kiếm
            x1 = max(0, initial_x - search_area)
            y1 = max(0, initial_y - search_area)
            x2 = min(image.shape[1], initial_x + search_area)
            y2 = min(image.shape[0], initial_y + search_area)

            search_region = image[int(y1):int(y2), int(x1):int(x2)]
            result = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)

            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > 0.65:
                # Vị trí tương đối trong search region
                rel_x, rel_y = max_loc[0], max_loc[1]

                # Tính tọa độ toàn cục
                found_x = x1 + rel_x + w / 2.0
                found_y = y1 + rel_y + h / 2.0

                # Nội suy sub-pixel bằng Gaussian fitting
                sub_pixel_offset_x, sub_pixel_offset_y = EnhancedROITracker._subpixel_refinement(result, max_loc)
                found_x = round(found_x + sub_pixel_offset_x, 8)
                found_y = round(found_y + sub_pixel_offset_y, 8)

                print("Update by template match")

                return found_x, found_y, round(max_val, 8)

            search_area *= 2

        return None
    
    @staticmethod
    def _feature_match(image, template, initial_x, initial_y, initial_search_area):
        try:
            # Ensure parameters are float
            initial_x, initial_y = float(initial_x), float(initial_y)
            initial_search_area = float(initial_search_area)

            # Validate image and template sizes
            if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
                template = cv2.resize(template, (image.shape[1], image.shape[0]), 
                                    interpolation=cv2.INTER_AREA)

            # Detectors configuration
            detectors = [
                cv2.ORB_create(nfeatures=500),
                cv2.AKAZE_create()
            ]

            for detector in detectors:
                try:
                    # Detect keypoints and descriptors
                    kp1, des1 = detector.detectAndCompute(template, None)
                    kp2, des2 = detector.detectAndCompute(image, None)

                    # Skip if insufficient keypoints
                    if des1 is None or des2 is None or len(kp1) == 0 or len(kp2) == 0:
                        continue

                    # Brute Force Matcher
                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = bf.match(des1, des2)
                    
                    # Filter matches near search area
                    good_matches = [
                        m for m in matches
                        if abs(kp2[m.trainIdx].pt[0] - initial_x) <= initial_search_area and
                        abs(kp2[m.trainIdx].pt[1] - initial_y) <= initial_search_area
                    ]

                    if good_matches:
                        # Best match selection
                        good_matches = sorted(good_matches, key=lambda x: x.distance)
                        best_match = good_matches[0]
                        
                        # Match coordinates
                        match_x = float(kp2[best_match.trainIdx].pt[0])
                        match_y = float(kp2[best_match.trainIdx].pt[1])

                        # Create dummy result for subpixel refinement
                        result = np.zeros((3,3), dtype=np.float32)
                        result[1,1] = 1.0  # Center point
                        
                        # Subpixel refinement
                        sub_x, sub_y = EnhancedROITracker._subpixel_refinement(result, (1,1))
                        
                        # Apply subpixel offset
                        found_x = round(match_x + sub_x, 8)
                        found_y = round(match_y + sub_y, 8)
                        
                        # Confidence calculation
                        confidence = round(max(0, 1 - (float(best_match.distance) / 300)), 8)
                        
                        print("Update by feature match")

                        return found_x, found_y, confidence

                except Exception as detector_error:
                    print(f"Detector error: {detector_error}")
                    continue

            # Fallback to template matching
            # return EnhancedROITracker._template_match(image, template, initial_x, initial_y, initial_search_area)

        except Exception as e:
            print(f"Feature match error: {e}")
            return None

    @staticmethod
    def _subpixel_refinement(result, max_loc):
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
        except Exception:
            offset_x_gaussian, offset_y_gaussian = 0, 0

        roi_size = 3
        roi_x_start = max(x - roi_size // 2, 0)
        roi_y_start = max(y - roi_size // 2, 0)
        roi_x_end = min(x + roi_size // 2 + 1, w)
        roi_y_end = min(y + roi_size // 2 + 1, h)

        roi = result[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

        moments = cv2.moments(roi)
        if moments["m00"] != 0:
            offset_x_moments = moments["m10"] / moments["m00"] - x
            offset_y_moments = moments["m01"] / moments["m00"] - y
        else:
            offset_x_moments, offset_y_moments = 0, 0

        weighted_sum = np.sum(roi)
        if weighted_sum > 0:
            grid_x, grid_y = np.meshgrid(
                np.arange(roi_x_start, roi_x_end),
                np.arange(roi_y_start, roi_y_end)
            )
            offset_x_weighted = (np.sum(grid_x * roi) / weighted_sum) - x
            offset_y_weighted = (np.sum(grid_y * roi) / weighted_sum) - y
        else:
            offset_x_weighted, offset_y_weighted = 0, 0

        if abs(offset_x_gaussian) <= 1 and abs(offset_y_gaussian) <= 1:
            return offset_x_gaussian, offset_y_gaussian
        elif abs(offset_x_moments) <= 1 and abs(offset_y_moments) <= 1:
            return offset_x_moments, offset_y_moments
        else:
            return offset_x_weighted, offset_y_weighted
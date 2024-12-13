import pypylon.pylon as pylon
import cv2
import numpy as np
from pyzbar.pyzbar import decode

class QRCornerTracker:
    def __init__(self):
        # Camera setup
        self.tlFactory = pylon.TlFactory.GetInstance()
        devices = self.tlFactory.EnumerateDevices()
        if len(devices) == 0:
            raise RuntimeError("No Basler camera found")
        
        self.camera = pylon.InstantCamera(self.tlFactory.CreateDevice(devices[0]))
        self.camera.Open()
        
        # Camera configuration
        self.camera.ExposureTime.SetValue(2000.0)
        self.camera.Width.SetValue(1000)
        self.camera.Height.SetValue(1000)
        self.camera.OffsetX.SetValue(711)
        self.camera.OffsetY.SetValue(549)
    
    def find_qr_top_right_corner(self, image):
        """Detect QR code and find its top-right corner"""
        # Decode QR codes
        qr_codes = decode(image)
        
        top_right_points = []
        for qr in qr_codes:
            # Get QR code polygon points
            polygon = qr.polygon
            
            # Find the bounding rectangle
            x, y, w, h = qr.rect
            
            # Sort points to find top-right corner
            # Sort polygon points by their x + y values to identify top-right
            sorted_points = sorted(polygon, key=lambda point: point.x + point.y)
            top_right = sorted_points[-1]
            
            top_right_points.append({
                'point': (top_right.x, top_right.y),
                'roi': (x, y, w, h)
            })
        
        return top_right_points
    
    def precise_corner_detection(self, image, roi):
        """More precise corner detection using image processing"""
        x, y, w, h = roi
        
        # Crop the ROI
        roi_image = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        gray = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the contour with the largest area (likely the QR code)
        max_contour = max(contours, key=cv2.contourArea)
        
        # Find convex hull to smooth out the contour
        hull = cv2.convexHull(max_contour, returnPoints=True)
        
        # Find the point with the maximum x + y coordinates
        top_right_local = max(hull, key=lambda point: point[0][0] + point[0][1])[0]
        
        # Adjust to global coordinates
        top_right_global = (top_right_local[0] + x, top_right_local[1] + y)
        
        return top_right_global
    
    def track_corners(self):
        """Track QR code corners"""
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        try:
            while self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                
                if grabResult.GrabSucceeded():
                    # Convert image to BGR if needed
                    image = grabResult.GetArray()
                    if len(image.shape) == 2:
                        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                    
                    # Detect QR codes
                    top_right_results = self.find_qr_top_right_corner(image)
                    
                    for result in top_right_results:
                        # Get initial top-right point
                        initial_point = result['point']
                        roi = result['roi']
                        
                        # Get more precise corner
                        precise_point = self.precise_corner_detection(image, roi)
                        
                        # Print coordinates
                        print(f"Initial Top-Right: {initial_point}")
                        print(f"Precise Top-Right: {precise_point}")
                        
                        # Draw initial point (blue)
                        cv2.circle(image, initial_point, 5, (255, 0, 0), -1)
                        
                        # Draw precise point (red)
                        cv2.circle(image, precise_point, 5, (0, 0, 255), -1)
                        
                        # Draw ROI rectangle
                        x, y, w, h = roi
                        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display image
                    cv2.namedWindow('QR Corner Tracking', cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('QR Corner Tracking', 800, 800)
                    cv2.imshow('QR Corner Tracking', image)
                    
                    # Exit if 'q' is pressed
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                grabResult.Release()
        
        except Exception as e:
            print(f"An error occurred: {e}")
        
        finally:
            self.camera.StopGrabbing()
            self.camera.Close()
            cv2.destroyAllWindows()

# Run tracking
if __name__ == "__main__":
    tracker = QRCornerTracker()
    tracker.track_corners()
# camera_module.py
from pypylon import pylon
import cv2
import numpy as np

class CameraManager:
    def __init__(self):
        self.cameras = []
        self.selected_camera = None

    def refresh_camera_list(self):
        """Lấy danh sách các camera kết nối"""
        tlFactory = pylon.TlFactory.GetInstance()
        self.cameras = tlFactory.EnumerateDevices()
        return [(cam.GetModelName(), cam.GetSerialNumber()) for cam in self.cameras]

    def select_camera(self, index):
        """Kết nối với camera được chọn"""
        if index < 0 or index >= len(self.cameras):
            raise ValueError("Index camera không hợp lệ")

        if self.selected_camera:
            self.selected_camera.Close()

        tlFactory = pylon.TlFactory.GetInstance()
        self.selected_camera = pylon.InstantCamera(tlFactory.CreateDevice(self.cameras[index]))
        self.selected_camera.Open()
        self.selected_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    def get_frame(self):
        """Lấy một khung hình từ camera"""
        if not self.selected_camera or not self.selected_camera.IsGrabbing():
            return None

        grab_result = self.selected_camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grab_result.GrabSucceeded():
            frame = grab_result.GetArray()
            grab_result.Release()

            # Chuyển đổi hệ màu nếu cần
            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return None

    def close_camera(self):
        """Đóng kết nối với camera"""
        if self.selected_camera:
            self.selected_camera.StopGrabbing()
            self.selected_camera.Close()

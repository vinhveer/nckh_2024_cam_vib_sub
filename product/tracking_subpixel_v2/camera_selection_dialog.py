from pypylon import pylon
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QDialogButtonBox, QMessageBox

class CameraSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose Camera to start")
        self.setGeometry(200, 200, 800, 150)
        
        layout = QVBoxLayout()
        
        # Camera selection dropdown
        self.camera_combo = QComboBox()
        layout.addWidget(QLabel("Available Cameras:"))
        layout.addWidget(self.camera_combo)
        
        # Populate camera list
        self.populate_camera_list()
        
        # Button box
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
        
        self.setLayout(layout)
        
        # If no cameras found, show error
        if self.camera_combo.count() == 0:
            QMessageBox.warning(self, "No Cameras", "No Basler cameras found.")
    
    def populate_camera_list(self):
        # Get list of available cameras
        camera_list = pylon.TlFactory.GetInstance().EnumerateDevices()
        
        for camera in camera_list:
            camera_info = f"{camera.GetSerialNumber()} - {camera.GetModelName()}"
            self.camera_combo.addItem(camera_info, userData=camera.GetSerialNumber())
    
    def get_selected_camera_serial(self):
        return self.camera_combo.currentData()
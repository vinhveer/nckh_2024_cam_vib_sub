import sys
from PyQt5.QtWidgets import QApplication

from basler_camera_roi_tracker import BaslerCameraROITracker

if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
        /* Global Font and Background */
        QWidget {
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            background-color: #f4f6f9;
            color: #2c3e50;
        }

        /* Buttons - Clean Design */
        QPushButton {
            font-size: 20px;
            padding: 12px 25px;
            background-color: #3498db;
            color: white;
            border-radius: 8px;
            border: none;
        }

        QPushButton:hover {
            background-color: #2980b9;
        }

        QPushButton:pressed {
            background-color: #2574a9;
        }

        QPushButton:disabled {
            background-color: #bdc3c7;
            color: #7f8c8d;
        }

        /* Labels - Clean and Professional */
        QLabel {
            font-size: 20px;
            color: #34495e;
            margin-bottom: 8px;
        }

        /* Combo Boxes - Sleek Dropdown */
        QComboBox {
            font-size: 20px;
            padding: 10px 15px;
            background-color: white;
            color: #2c3e50;
            border: 1px solid #e0e4e8;
            border-radius: 6px;
            min-height: 40px;
        }

        QComboBox QAbstractItemView {
            font-size: 20px;
            background-color: white;
            color: #2c3e50;
            selection-background-color: #3498db;
            selection-color: white;
        }

        QComboBox QAbstractItemView::item {
            min-height: 35px;
            padding: 5px 10px;
        }

        QComboBox QAbstractItemView::item:selected {
            background-color: #3498db;
            color: white;
        }

        /* Optional: Text Inputs */
        QLineEdit {
            font-size: 20px;
            padding: 10px;
            border: 1px solid #e0e4e8;
            border-radius: 6px;
            background-color: white;
            color: #2c3e50;
        }

        QLineEdit:focus {
            border: 1px solid #3498db;
        }
    """)


    window = BaslerCameraROITracker()
    window.show()
    sys.exit(app.exec_())

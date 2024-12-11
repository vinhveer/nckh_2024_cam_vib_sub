# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'untitled.ui'
# Created by: PyQt5 UI code generator 5.15.9
# WARNING: Do not manually edit this file unless necessary.

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        # Main Window Configuration
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(911, 579)

        # Central Widget
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # Stacked Widget
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        self.stackedWidget.setGeometry(QtCore.QRect(10, 10, 891, 521))
        self.stackedWidget.setObjectName("stackedWidget")

        # Page: Tracking Realtime
        self.pageTrackingRealtime = QtWidgets.QWidget()
        self.pageTrackingRealtime.setObjectName("pageTrackingRealtime")

        # Layout and Buttons for Tracking Realtime
        self.gridLayoutWidget = QtWidgets.QWidget(self.pageTrackingRealtime)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(0, 120, 891, 401))
        self.displayFrame = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.displayFrame.setContentsMargins(0, 0, 0, 0)

        self.buttonSetCameraFrameResolution = QtWidgets.QPushButton("Set Resolution", self.pageTrackingRealtime)
        self.buttonSetCameraFrameResolution.setGeometry(QtCore.QRect(0, 0, 141, 28))

        self.buttonSelectROICamera = QtWidgets.QPushButton("Capture and Select ROI", self.pageTrackingRealtime)
        self.buttonSelectROICamera.setGeometry(QtCore.QRect(150, 0, 181, 28))

        self.buttonStartTrackingCamera = QtWidgets.QPushButton("Start", self.pageTrackingRealtime)
        self.buttonStartTrackingCamera.setGeometry(QtCore.QRect(340, 0, 93, 28))

        self.buttonStopTrackingCamera = QtWidgets.QPushButton("Stop and Show Statics", self.pageTrackingRealtime)
        self.buttonStopTrackingCamera.setGeometry(QtCore.QRect(440, 0, 181, 28))

        self.textBrowser = QtWidgets.QTextBrowser(self.pageTrackingRealtime)
        self.textBrowser.setGeometry(QtCore.QRect(0, 40, 651, 71))

        self.comboBoxChooseCamera = QtWidgets.QComboBox(self.pageTrackingRealtime)
        self.comboBoxChooseCamera.setGeometry(QtCore.QRect(660, 40, 231, 25))

        self.label = QtWidgets.QLabel("Choose Camera", self.pageTrackingRealtime)
        self.label.setGeometry(QtCore.QRect(660, 10, 121, 19))

        self.buttonApplySettingsCamera = QtWidgets.QPushButton("Apply Settings", self.pageTrackingRealtime)
        self.buttonApplySettingsCamera.setGeometry(QtCore.QRect(660, 70, 121, 28))

        self.stackedWidget.addWidget(self.pageTrackingRealtime)

        # Page: Tracking Video
        self.pageTrackingVideo = QtWidgets.QWidget()
        self.pageTrackingVideo.setObjectName("pageTrackingVideo")

        self.buttonChooseVideo = QtWidgets.QPushButton("Choose Videos ...", self.pageTrackingVideo)
        self.buttonChooseVideo.setGeometry(QtCore.QRect(0, 0, 151, 28))

        self.buttonSetResolutionTrackingVideo = QtWidgets.QPushButton("Set Resolution", self.pageTrackingVideo)
        self.buttonSetResolutionTrackingVideo.setGeometry(QtCore.QRect(270, 0, 141, 28))

        self.buttonSelectROIVideo = QtWidgets.QPushButton("Capture and Select ROI", self.pageTrackingVideo)
        self.buttonSelectROIVideo.setGeometry(QtCore.QRect(420, 0, 181, 28))

        self.pushStartTrackingVideo = QtWidgets.QPushButton("Start", self.pageTrackingVideo)
        self.pushStartTrackingVideo.setGeometry(QtCore.QRect(610, 0, 93, 28))

        self.buttonStopTrackingVideo = QtWidgets.QPushButton("Stop and Show Statics", self.pageTrackingVideo)
        self.buttonStopTrackingVideo.setGeometry(QtCore.QRect(710, 0, 181, 28))

        self.textBrowserTrackingVideo = QtWidgets.QTextBrowser(self.pageTrackingVideo)
        self.textBrowserTrackingVideo.setGeometry(QtCore.QRect(0, 40, 891, 71))

        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.pageTrackingVideo)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(0, 120, 891, 401))
        self.displayFrameVideo = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.displayFrameVideo.setContentsMargins(0, 0, 0, 0)

        self.stackedWidget.addWidget(self.pageTrackingVideo)

        # Page: Camera Settings
        self.pageCameraSettings = QtWidgets.QWidget()
        self.pageCameraSettings.setObjectName("pageCameraSettings")

        self.label_2 = QtWidgets.QLabel("Basler Camera Settings", self.pageCameraSettings)
        self.label_2.setGeometry(QtCore.QRect(0, 0, 571, 51))
        font = QtGui.QFont("Segoe UI", 18, QtGui.QFont.Bold)
        self.label_2.setFont(font)

        self.sliderXResolution = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.pageCameraSettings)
        self.sliderXResolution.setGeometry(QtCore.QRect(0, 90, 881, 22))
        self.label_3 = QtWidgets.QLabel("X resolution", self.pageCameraSettings)
        self.label_3.setGeometry(QtCore.QRect(0, 60, 161, 19))

        self.sliderYResolution = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.pageCameraSettings)
        self.sliderYResolution.setGeometry(QtCore.QRect(0, 150, 881, 22))
        self.label_4 = QtWidgets.QLabel("Y resolution", self.pageCameraSettings)
        self.label_4.setGeometry(QtCore.QRect(0, 120, 161, 19))

        self.sliderExposureTime = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.pageCameraSettings)
        self.sliderExposureTime.setGeometry(QtCore.QRect(0, 210, 881, 22))
        self.label_5 = QtWidgets.QLabel("ExposureTime", self.pageCameraSettings)
        self.label_5.setGeometry(QtCore.QRect(0, 180, 161, 19))

        self.sliderAcquisitionFrameRate = QtWidgets.QSlider(QtCore.Qt.Horizontal, self.pageCameraSettings)
        self.sliderAcquisitionFrameRate.setGeometry(QtCore.QRect(0, 270, 881, 22))
        self.label_6 = QtWidgets.QLabel("Acquisition Frame Rate", self.pageCameraSettings)
        self.label_6.setGeometry(QtCore.QRect(0, 240, 251, 19))

        self.stackedWidget.addWidget(self.pageCameraSettings)

        # Main Window Setup
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        MainWindow.setMenuBar(self.menubar)

        # Tạo menu chính
        self.menuFile = QtWidgets.QMenu("File", self.menubar)
        self.menuSelectTrackingMode = QtWidgets.QMenu("Select Tracking Mode", self.menubar)

        # Thêm menu chính vào menubar
        self.menubar.addMenu(self.menuFile)
        self.menubar.addMenu(self.menuSelectTrackingMode)

        # Tạo QAction cho mục File
        self.actionCameraSettings = QtWidgets.QAction("Camera Settings", MainWindow)
        self.actionExit = QtWidgets.QAction("Exit", MainWindow)

        # Thêm QAction vào menu File
        self.menuFile.addAction(self.actionCameraSettings)
        self.menuFile.addAction(self.actionExit)

        # Tạo QAction cho mục Select Tracking Mode
        self.actionTrackingRealtime = QtWidgets.QAction("Tracking Realtime", MainWindow)
        self.actionTrackingVideo = QtWidgets.QAction("Tracking Video", MainWindow)

        # Thêm QAction vào menu Select Tracking Mode
        self.menuSelectTrackingMode.addAction(self.actionTrackingRealtime)
        self.menuSelectTrackingMode.addAction(self.actionTrackingVideo)

        # Liên kết QAction với các hành động tương ứng
        self.actionCameraSettings.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(2))
        self.actionTrackingRealtime.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(0))
        self.actionTrackingVideo.triggered.connect(lambda: self.stackedWidget.setCurrentIndex(1))
        self.actionExit.triggered.connect(QtWidgets.qApp.quit)  # Đóng ứng dụng

        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        # Translate and Default Setup
        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle("Tracking ROI")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

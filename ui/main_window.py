import os
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QAction, QVBoxLayout, QWidget, QLabel, QPushButton
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from modules.model_loader import load_model
from modules.dicom_loader import load_dicom
from modules.visualization import add_axes, add_bounding_box, add_lighting
from modules.interaction import set_interaction_style
from modules.annotation import create_annotation
from modules.ai_analysis import analyze_dicom
from modules.cloud_integration import upload_to_s3, upload_to_firebase

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Medical 3D Viewer")
        self.setGeometry(100, 100, 1000, 700)

        # Create the central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # Layout
        layout = QVBoxLayout()
        self.central_widget.setLayout(layout)

        # VTK Widget (3D Viewer)
        self.vtk_widget = QVTKRenderWindowInteractor(self.central_widget)
        layout.addWidget(self.vtk_widget)

        # VTK Renderer & Interactor
        self.renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Set Interaction Style
        set_interaction_style(self.interactor)

        # Add Axes & Bounding Box
        add_axes(self.interactor)
        add_bounding_box(self.renderer)
        add_lighting(self.renderer)

        # Menu Bar
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        open_action = QAction("Open 3D File", self)
        open_action.triggered.connect(self.load_3d_file)
        file_menu.addAction(open_action)

        upload_menu = menu_bar.addMenu("Upload")
        upload_s3_action = QAction("Upload to AWS S3", self)
        upload_s3_action.triggered.connect(self.upload_s3)
        upload_menu.addAction(upload_s3_action)

        upload_firebase_action = QAction("Upload to Firebase", self)
        upload_firebase_action.triggered.connect(self.upload_firebase)
        upload_menu.addAction(upload_firebase_action)

        # Status Bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Start VTK
        self.interactor.Initialize()
        self.show()

    def load_3d_file(self):
        """ Open STL/OBJ/DICOM file and display it """
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open 3D Model", "", 
                                                   "3D Files (*.stl *.obj *.dcm);;All Files (*)", options=options)

        if file_path:
            self.status_label.setText(f"Loading: {os.path.basename(file_path)}")
            if file_path.endswith(".dcm"):
                self.display_dicom(file_path)
            else:
                self.display_3d_model(file_path)

    def display_3d_model(self, file_path):
        """ Load and display STL/OBJ models """
        actor = load_model(file_path)
        self.renderer.RemoveAllViewProps()
        add_axes(self.interactor)
        add_bounding_box(self.renderer)
        self.renderer.AddActor(actor)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.status_label.setText("Model Loaded Successfully")

    def display_dicom(self, file_path):
        """ Load and display DICOM images """
        volume = load_dicom(file_path)
        self.renderer.RemoveAllViewProps()
        add_axes(self.interactor)
        add_bounding_box(self.renderer)
        self.renderer.AddVolume(volume)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()
        self.status_label.setText("DICOM Loaded Successfully")

        # Run AI Analysis
        result = analyze_dicom(file_path)
        self.status_label.setText(result)

    def upload_s3(self):
        """ Upload selected file to AWS S3 """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File to Upload", "", "All Files (*)")
        if file_path:
            url = upload_to_s3(file_path, "your-s3-bucket-name", os.path.basename(file_path))
            self.status_label.setText(f"Uploaded to S3: {url}")

    def upload_firebase(self):
        """ Upload selected file to Firebase """
        file_path, _ = QFileDialog.getOpenFileName(self, "Select File to Upload", "", "All Files (*)")
        if file_path:
            url = upload_to_firebase(file_path, "your-firebase-bucket-name")
            self.status_label.setText(f"Uploaded to Firebase: {url}")

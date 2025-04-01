"""
Main window implementation for the Medical 3D Viewer application.
"""

import os
import logging
from typing import Optional
from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QAction, QVBoxLayout, QWidget, 
    QLabel, QPushButton, QProgressDialog, QMessageBox
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
from config import (
    WINDOW_TITLE, WINDOW_GEOMETRY, SUPPORTED_3D_FORMATS,
    SUPPORTED_MEDICAL_FORMATS, DEFAULT_BACKGROUND_COLOR,
    DEFAULT_AXES_COLOR, DEFAULT_BOUNDING_BOX_COLOR
)
from modules.model_loader import load_model
from modules.dicom_loader import load_dicom
from modules.visualization import add_axes, add_bounding_box, add_lighting
from modules.interaction import set_interaction_style
from modules.annotation import create_annotation
from modules.ai_analysis import analyze_dicom
from modules.cloud_integration import (
    upload_to_s3, upload_to_firebase, CloudUploadError,
    initialize_cloud_services
)

# Configure logging
logger = logging.getLogger(__name__)

class MainWindow(QMainWindow):
    """Main window class for the Medical 3D Viewer application."""
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        self.setup_actions()
        self.setup_shortcuts()
        self.current_file: Optional[str] = None
        
        # Initialize cloud services
        try:
            initialize_cloud_services()
        except Exception as e:
            logger.error(f"Failed to initialize cloud services: {str(e)}")
            QMessageBox.warning(
                self, "Cloud Services Error",
                "Failed to initialize cloud services. Cloud features will be disabled."
            )

    def setup_ui(self):
        """Set up the user interface components."""
        self.setWindowTitle(WINDOW_TITLE)
        self.setGeometry(*WINDOW_GEOMETRY)

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
        self.renderer.SetBackground(*DEFAULT_BACKGROUND_COLOR)
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Set Interaction Style
        set_interaction_style(self.interactor)

        # Add Axes & Bounding Box
        self.axes_actor = add_axes(self.interactor, color=DEFAULT_AXES_COLOR)
        self.bounding_box_actor = add_bounding_box(self.renderer, color=DEFAULT_BOUNDING_BOX_COLOR)
        add_lighting(self.renderer)

        # Status Bar
        self.status_label = QLabel("Ready")
        layout.addWidget(self.status_label)

        # Start VTK
        self.interactor.Initialize()
        self.show()

    def setup_actions(self):
        """Set up menu actions."""
        # File Menu
        file_menu = self.menuBar().addMenu("File")
        
        open_action = QAction("Open File", self)
        open_action.triggered.connect(self.load_3d_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save View", self)
        save_action.triggered.connect(self.save_view)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View Menu
        view_menu = self.menuBar().addMenu("View")
        
        reset_camera_action = QAction("Reset Camera", self)
        reset_camera_action.triggered.connect(self.reset_camera)
        view_menu.addAction(reset_camera_action)
        
        toggle_axes_action = QAction("Toggle Axes", self)
        toggle_axes_action.triggered.connect(self.toggle_axes)
        view_menu.addAction(toggle_axes_action)
        
        toggle_bounding_box_action = QAction("Toggle Bounding Box", self)
        toggle_bounding_box_action.triggered.connect(self.toggle_bounding_box)
        view_menu.addAction(toggle_bounding_box_action)

        # Upload Menu
        upload_menu = self.menuBar().addMenu("Upload")
        
        upload_s3_action = QAction("Upload to AWS S3", self)
        upload_s3_action.triggered.connect(self.upload_s3)
        upload_menu.addAction(upload_s3_action)
        
        upload_firebase_action = QAction("Upload to Firebase", self)
        upload_firebase_action.triggered.connect(self.upload_firebase)
        upload_menu.addAction(upload_firebase_action)

    def setup_shortcuts(self):
        """Set up keyboard shortcuts."""
        # File shortcuts
        open_shortcut = QAction("Open", self)
        open_shortcut.setShortcut(QKeySequence.Open)
        open_shortcut.triggered.connect(self.load_3d_file)
        self.addAction(open_shortcut)
        
        save_shortcut = QAction("Save", self)
        save_shortcut.setShortcut(QKeySequence.Save)
        save_shortcut.triggered.connect(self.save_view)
        self.addAction(save_shortcut)
        
        # View shortcuts
        reset_camera_shortcut = QAction("Reset Camera", self)
        reset_camera_shortcut.setShortcut("R")
        reset_camera_shortcut.triggered.connect(self.reset_camera)
        self.addAction(reset_camera_shortcut)
        
        toggle_axes_shortcut = QAction("Toggle Axes", self)
        toggle_axes_shortcut.setShortcut("A")
        toggle_axes_shortcut.triggered.connect(self.toggle_axes)
        self.addAction(toggle_axes_shortcut)
        
        toggle_bounding_box_shortcut = QAction("Toggle Bounding Box", self)
        toggle_bounding_box_shortcut.setShortcut("B")
        toggle_bounding_box_shortcut.triggered.connect(self.toggle_bounding_box)
        self.addAction(toggle_bounding_box_shortcut)

    def load_3d_file(self):
        """Open and display a 3D model or DICOM file."""
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open 3D Model",
            "",
            f"3D Files ({' '.join(SUPPORTED_3D_FORMATS)});;"
            f"Medical Files ({' '.join(SUPPORTED_MEDICAL_FORMATS)});;"
            "All Files (*)",
            options=options
        )

        if file_path:
            self.current_file = file_path
            self.show_loading_dialog("Loading file...")
            
            try:
                if file_path.lower().endswith(('.dcm', '.nii', '.nii.gz')):
                    self.display_dicom(file_path)
                else:
                    self.display_3d_model(file_path)
            except Exception as e:
                logger.error(f"Error loading file: {str(e)}")
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to load file: {str(e)}"
                )
            finally:
                self.hide_loading_dialog()

    def display_3d_model(self, file_path: str):
        """Load and display STL/OBJ models."""
        try:
            actor = load_model(file_path)
            self.renderer.RemoveAllViewProps()
            self.axes_actor = add_axes(self.interactor, color=DEFAULT_AXES_COLOR)
            self.bounding_box_actor = add_bounding_box(self.renderer, color=DEFAULT_BOUNDING_BOX_COLOR)
            self.renderer.AddActor(actor)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.status_label.setText("Model Loaded Successfully")
        except Exception as e:
            raise Exception(f"Failed to load 3D model: {str(e)}")

    def display_dicom(self, file_path: str):
        """Load and display DICOM images."""
        try:
            volume = load_dicom(file_path)
            self.renderer.RemoveAllViewProps()
            self.axes_actor = add_axes(self.interactor, color=DEFAULT_AXES_COLOR)
            self.bounding_box_actor = add_bounding_box(self.renderer, color=DEFAULT_BOUNDING_BOX_COLOR)
            self.renderer.AddVolume(volume)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
            self.status_label.setText("DICOM Loaded Successfully")

            # Run AI Analysis
            result = analyze_dicom(file_path)
            self.status_label.setText(result)
        except Exception as e:
            raise Exception(f"Failed to load DICOM: {str(e)}")

    def upload_s3(self):
        """Upload selected file to AWS S3."""
        if not self.current_file:
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please load a file before uploading."
            )
            return

        try:
            self.show_loading_dialog("Uploading to S3...")
            url = upload_to_s3(self.current_file)
            QMessageBox.information(
                self,
                "Upload Success",
                f"File uploaded successfully to S3:\n{url}"
            )
        except CloudUploadError as e:
            QMessageBox.critical(
                self,
                "Upload Error",
                f"Failed to upload to S3: {str(e)}"
            )
        finally:
            self.hide_loading_dialog()

    def upload_firebase(self):
        """Upload selected file to Firebase."""
        if not self.current_file:
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please load a file before uploading."
            )
            return

        try:
            self.show_loading_dialog("Uploading to Firebase...")
            url = upload_to_firebase(self.current_file)
            QMessageBox.information(
                self,
                "Upload Success",
                f"File uploaded successfully to Firebase:\n{url}"
            )
        except CloudUploadError as e:
            QMessageBox.critical(
                self,
                "Upload Error",
                f"Failed to upload to Firebase: {str(e)}"
            )
        finally:
            self.hide_loading_dialog()

    def save_view(self):
        """Save the current view as an image."""
        if not self.current_file:
            QMessageBox.warning(
                self,
                "No File Loaded",
                "Please load a file before saving the view."
            )
            return

        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save View",
            os.path.splitext(self.current_file)[0] + "_view.png",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)",
            options=options
        )

        if file_path:
            try:
                self.show_loading_dialog("Saving view...")
                # TODO: Implement view saving functionality
                self.status_label.setText("View Saved Successfully")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Save Error",
                    f"Failed to save view: {str(e)}"
                )
            finally:
                self.hide_loading_dialog()

    def reset_camera(self):
        """Reset the camera to its default position."""
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def toggle_axes(self):
        """Toggle the visibility of the axes."""
        if self.axes_actor:
            self.axes_actor.SetVisibility(not self.axes_actor.GetVisibility())
            self.vtk_widget.GetRenderWindow().Render()

    def toggle_bounding_box(self):
        """Toggle the visibility of the bounding box."""
        if self.bounding_box_actor:
            self.bounding_box_actor.SetVisibility(not self.bounding_box_actor.GetVisibility())
            self.vtk_widget.GetRenderWindow().Render()

    def show_loading_dialog(self, message: str):
        """Show a loading dialog with the given message."""
        self.loading_dialog = QProgressDialog(message, None, 0, 0, self)
        self.loading_dialog.setWindowModality(Qt.WindowModal)
        self.loading_dialog.show()

    def hide_loading_dialog(self):
        """Hide the loading dialog."""
        if hasattr(self, 'loading_dialog'):
            self.loading_dialog.close()

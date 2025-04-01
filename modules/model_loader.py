"""
Model loader module for the Medical 3D Viewer application.
Provides functions for loading and processing 3D model files.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any
import vtk
from vtkmodules.vtkCommonCore import vtkObject
from vtkmodules.vtkRenderingCore import vtkActor, vtkProperty

# Configure logging
logger = logging.getLogger(__name__)

class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass

def load_model(
    file_path: str,
    color: Optional[Tuple[float, float, float]] = None,
    opacity: float = 1.0,
    interpolation: str = "phong",
    edge_visibility: bool = False,
    edge_color: Optional[Tuple[float, float, float]] = None,
    edge_width: float = 1.0
) -> vtkActor:
    """
    Load a 3D model file (STL or OBJ) and return a VTK actor.
    
    Args:
        file_path: Path to the model file
        color: RGB color tuple for the model (default: None)
        opacity: Opacity of the model (0.0 to 1.0, default: 1.0)
        interpolation: Shading interpolation method ("flat", "gouraud", "phong", default: "phong")
        edge_visibility: Whether to show edges (default: False)
        edge_color: RGB color tuple for edges (default: None)
        edge_width: Width of edges in pixels (default: 1.0)
        
    Returns:
        The VTK actor containing the loaded model
        
    Raises:
        ModelLoadError: If file cannot be loaded or parameters are invalid
    """
    if not os.path.exists(file_path):
        raise ModelLoadError(f"File not found: {file_path}")
        
    if not 0 <= opacity <= 1:
        raise ValueError("Opacity must be between 0 and 1")
        
    if interpolation not in ["flat", "gouraud", "phong"]:
        raise ValueError("Invalid interpolation method")
        
    try:
        # Select appropriate reader based on file extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".stl":
            reader = vtk.vtkSTLReader()
        elif ext == ".obj":
            reader = vtk.vtkOBJReader()
        else:
            raise ModelLoadError(f"Unsupported file format: {ext}")
            
        logger.info(f"Loading model from {file_path}")
        reader.SetFileName(file_path)
        reader.Update()
        
        # Create mapper
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        
        # Create actor
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        
        # Set material properties
        prop = actor.GetProperty()
        
        # Set interpolation
        if interpolation == "flat":
            prop.SetInterpolationToFlat()
        elif interpolation == "gouraud":
            prop.SetInterpolationToGouraud()
        else:  # phong
            prop.SetInterpolationToPhong()
            
        # Set color if provided
        if color:
            prop.SetColor(*color)
            
        # Set opacity
        prop.SetOpacity(opacity)
        
        # Set edge properties if enabled
        if edge_visibility:
            prop.EdgeVisibilityOn()
            if edge_color:
                prop.SetEdgeColor(*edge_color)
            prop.SetEdgeWidth(edge_width)
            
        logger.debug("Model loaded successfully")
        return actor
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise ModelLoadError(f"Failed to load model: {str(e)}")

def get_model_info(file_path: str) -> Dict[str, Any]:
    """
    Get information about a 3D model file.
    
    Args:
        file_path: Path to the model file
        
    Returns:
        Dictionary containing model information
        
    Raises:
        ModelLoadError: If file cannot be loaded
    """
    if not os.path.exists(file_path):
        raise ModelLoadError(f"File not found: {file_path}")
        
    try:
        # Load the model
        actor = load_model(file_path)
        mapper = actor.GetMapper()
        polydata = mapper.GetInput()
        
        # Get bounds
        bounds = polydata.GetBounds()
        
        # Get number of points and cells
        num_points = polydata.GetNumberOfPoints()
        num_cells = polydata.GetNumberOfCells()
        
        # Get surface area and volume
        mass = vtk.vtkMassProperties()
        mass.SetInputData(polydata)
        mass.Update()
        
        info = {
            "file_path": file_path,
            "file_size": os.path.getsize(file_path),
            "bounds": bounds,
            "num_points": num_points,
            "num_cells": num_cells,
            "surface_area": mass.GetSurfaceArea(),
            "volume": mass.GetVolume(),
            "center_of_mass": mass.GetCenterOfMass()
        }
        
        logger.debug(f"Retrieved model info: {info}")
        return info
        
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise ModelLoadError(f"Failed to get model info: {str(e)}")

def center_model(actor: vtkActor) -> None:
    """
    Center a model actor at the origin.
    
    Args:
        actor: The VTK actor to center
        
    Raises:
        ValueError: If actor is invalid
    """
    if not isinstance(actor, vtkActor):
        raise ValueError("Invalid actor provided")
        
    try:
        mapper = actor.GetMapper()
        polydata = mapper.GetInput()
        
        # Get bounds
        bounds = polydata.GetBounds()
        
        # Calculate center
        center = [
            (bounds[0] + bounds[1]) / 2,
            (bounds[2] + bounds[3]) / 2,
            (bounds[4] + bounds[5]) / 2
        ]
        
        # Create transform
        transform = vtk.vtkTransform()
        transform.Translate(-center[0], -center[1], -center[2])
        
        # Apply transform
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(polydata)
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        
        # Update mapper input
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        
        logger.debug("Model centered successfully")
    except Exception as e:
        logger.error(f"Failed to center model: {str(e)}")
        raise ModelLoadError(f"Failed to center model: {str(e)}")

def normalize_model(actor: vtkActor) -> None:
    """
    Normalize a model actor to fit within a unit cube.
    
    Args:
        actor: The VTK actor to normalize
        
    Raises:
        ValueError: If actor is invalid
    """
    if not isinstance(actor, vtkActor):
        raise ValueError("Invalid actor provided")
        
    try:
        mapper = actor.GetMapper()
        polydata = mapper.GetInput()
        
        # Get bounds
        bounds = polydata.GetBounds()
        
        # Calculate scale factors
        scale = max(
            bounds[1] - bounds[0],
            bounds[3] - bounds[2],
            bounds[5] - bounds[4]
        )
        
        # Create transform
        transform = vtk.vtkTransform()
        transform.Scale(1/scale, 1/scale, 1/scale)
        
        # Apply transform
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputData(polydata)
        transform_filter.SetTransform(transform)
        transform_filter.Update()
        
        # Update mapper input
        mapper.SetInputConnection(transform_filter.GetOutputPort())
        
        logger.debug("Model normalized successfully")
    except Exception as e:
        logger.error(f"Failed to normalize model: {str(e)}")
        raise ModelLoadError(f"Failed to normalize model: {str(e)}")

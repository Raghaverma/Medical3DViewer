"""
Visualization module for the Medical 3D Viewer application.
Provides functions for adding various visualization elements to the 3D scene.
"""

import logging
from typing import Tuple, Optional, Union
import vtk
from vtkmodules.vtkCommonCore import vtkObject

# Configure logging
logger = logging.getLogger(__name__)

def add_axes(
    interactor: vtk.vtkRenderWindowInteractor,
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    size: float = 0.2
) -> vtk.vtkOrientationMarkerWidget:
    """
    Add an orientation axes widget to the scene.
    
    Args:
        interactor: The VTK interactor to attach the widget to
        color: RGB color tuple for the axes (default: white)
        size: Size of the widget in normalized viewport coordinates (default: 0.2)
        
    Returns:
        The created orientation marker widget
        
    Raises:
        ValueError: If interactor is invalid or size is out of range
    """
    if not isinstance(interactor, vtk.vtkRenderWindowInteractor):
        raise ValueError("Invalid interactor provided")
    if not 0 < size <= 1:
        raise ValueError("Size must be between 0 and 1")
        
    try:
        axes = vtk.vtkAxesActor()
        widget = vtk.vtkOrientationMarkerWidget()
        widget.SetOutlineColor(*color)
        widget.SetOrientationMarker(axes)
        widget.SetInteractor(interactor)
        widget.SetViewport(0.0, 0.0, size, size)
        widget.SetEnabled(True)
        widget.InteractiveOff()
        
        logger.debug("Added orientation axes widget to scene")
        return widget
    except Exception as e:
        logger.error(f"Failed to add axes: {str(e)}")
        raise

def add_bounding_box(
    renderer: vtk.vtkRenderer,
    color: Tuple[float, float, float] = (0.8, 0.8, 0.8),
    grid_lines: bool = True
) -> vtk.vtkCubeAxesActor:
    """
    Add a bounding box to the scene.
    
    Args:
        renderer: The VTK renderer to add the bounding box to
        color: RGB color tuple for the bounding box (default: light gray)
        grid_lines: Whether to show grid lines (default: True)
        
    Returns:
        The created cube axes actor
        
    Raises:
        ValueError: If renderer is invalid
    """
    if not isinstance(renderer, vtk.vtkRenderer):
        raise ValueError("Invalid renderer provided")
        
    try:
        cube_axes = vtk.vtkCubeAxesActor()
        cube_axes.SetCamera(renderer.GetActiveCamera())
        
        if grid_lines:
            cube_axes.DrawXGridlinesOn()
            cube_axes.DrawYGridlinesOn()
            cube_axes.DrawZGridlinesOn()
        else:
            cube_axes.DrawXGridlinesOff()
            cube_axes.DrawYGridlinesOff()
            cube_axes.DrawZGridlinesOff()
            
        # Set colors
        cube_axes.GetXAxesLinesProperty().SetColor(*color)
        cube_axes.GetYAxesLinesProperty().SetColor(*color)
        cube_axes.GetZAxesLinesProperty().SetColor(*color)
        
        renderer.AddActor(cube_axes)
        logger.debug("Added bounding box to scene")
        return cube_axes
    except Exception as e:
        logger.error(f"Failed to add bounding box: {str(e)}")
        raise

def add_lighting(
    renderer: vtk.vtkRenderer,
    position: Tuple[float, float, float] = (100, 100, 100),
    intensity: float = 1.0,
    ambient: float = 0.3
) -> vtk.vtkLight:
    """
    Add lighting to the scene for better visualization.
    
    Args:
        renderer: The VTK renderer to add lighting to
        position: Position of the light source (default: (100, 100, 100))
        intensity: Light intensity (default: 1.0)
        ambient: Ambient light intensity (default: 0.3)
        
    Returns:
        The created light source
        
    Raises:
        ValueError: If renderer is invalid or intensity/ambient are out of range
    """
    if not isinstance(renderer, vtk.vtkRenderer):
        raise ValueError("Invalid renderer provided")
    if not 0 <= intensity <= 1:
        raise ValueError("Intensity must be between 0 and 1")
    if not 0 <= ambient <= 1:
        raise ValueError("Ambient must be between 0 and 1")
        
    try:
        light = vtk.vtkLight()
        light.SetFocalPoint(0, 0, 0)
        light.SetPosition(*position)
        light.SetIntensity(intensity)
        light.SetAmbientColor(ambient, ambient, ambient)
        renderer.AddLight(light)
        
        logger.debug("Added lighting to scene")
        return light
    except Exception as e:
        logger.error(f"Failed to add lighting: {str(e)}")
        raise

def set_background_color(
    renderer: vtk.vtkRenderer,
    color: Tuple[float, float, float]
) -> None:
    """
    Set the background color of the renderer.
    
    Args:
        renderer: The VTK renderer to set the background color for
        color: RGB color tuple for the background
        
    Raises:
        ValueError: If renderer is invalid or color values are out of range
    """
    if not isinstance(renderer, vtk.vtkRenderer):
        raise ValueError("Invalid renderer provided")
    if not all(0 <= c <= 1 for c in color):
        raise ValueError("Color values must be between 0 and 1")
        
    try:
        renderer.SetBackground(*color)
        logger.debug(f"Set background color to {color}")
    except Exception as e:
        logger.error(f"Failed to set background color: {str(e)}")
        raise

def add_text_overlay(
    renderer: vtk.vtkRenderer,
    text: str,
    position: Tuple[float, float] = (0.02, 0.95),
    color: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    font_size: int = 14
) -> vtk.vtkTextActor:
    """
    Add text overlay to the scene.
    
    Args:
        renderer: The VTK renderer to add the text to
        text: The text to display
        position: Position of the text in normalized viewport coordinates (default: top-left)
        color: RGB color tuple for the text (default: white)
        font_size: Font size in pixels (default: 14)
        
    Returns:
        The created text actor
        
    Raises:
        ValueError: If renderer is invalid or position is out of range
    """
    if not isinstance(renderer, vtk.vtkRenderer):
        raise ValueError("Invalid renderer provided")
    if not all(0 <= p <= 1 for p in position):
        raise ValueError("Position values must be between 0 and 1")
        
    try:
        text_actor = vtk.vtkTextActor()
        text_actor.SetInput(text)
        text_actor.GetTextProperty().SetColor(*color)
        text_actor.GetTextProperty().SetFontSize(font_size)
        text_actor.SetPosition(*position)
        
        renderer.AddActor2D(text_actor)
        logger.debug(f"Added text overlay: {text}")
        return text_actor
    except Exception as e:
        logger.error(f"Failed to add text overlay: {str(e)}")
        raise

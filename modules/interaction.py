"""
Interaction module for the Medical 3D Viewer application.
Provides functions for handling user interactions with the 3D scene.
"""

import logging
from typing import List, Tuple, Optional, Union
import vtk
from vtkmodules.vtkCommonCore import vtkObject
from vtkmodules.vtkRenderingCore import vtkActor, vtkVolume

# Configure logging
logger = logging.getLogger(__name__)

class InteractionError(Exception):
    """Custom exception for interaction-related errors."""
    pass

def set_interaction_style(
    interactor: vtk.vtkRenderWindowInteractor,
    style_type: str = "trackball"
) -> None:
    """
    Set the interaction style for the VTK interactor.
    
    Args:
        interactor: The VTK interactor to set the style for
        style_type: Type of interaction style ("trackball", "joystick", "flight", "image")
        
    Raises:
        ValueError: If interactor is invalid or style_type is not supported
    """
    if not isinstance(interactor, vtk.vtkRenderWindowInteractor):
        raise ValueError("Invalid interactor provided")
        
    style_map = {
        "trackball": vtk.vtkInteractorStyleTrackballCamera,
        "joystick": vtk.vtkInteractorStyleJoystickCamera,
        "flight": vtk.vtkInteractorStyleFlight,
        "image": vtk.vtkInteractorStyleImage
    }
    
    if style_type not in style_map:
        raise ValueError(f"Unsupported style type: {style_type}")
        
    try:
        style = style_map[style_type]()
        interactor.SetInteractorStyle(style)
        logger.debug(f"Set interaction style to {style_type}")
    except Exception as e:
        logger.error(f"Failed to set interaction style: {str(e)}")
        raise InteractionError(f"Failed to set interaction style: {str(e)}")

def add_clipping_planes(
    actor: Union[vtkActor, vtkVolume],
    origin: Optional[Tuple[float, float, float]] = None,
    normals: Optional[List[Tuple[float, float, float]]] = None
) -> vtk.vtkPlaneCollection:
    """
    Add clipping planes to an actor for better visualization.
    
    Args:
        actor: The VTK actor or volume to add clipping planes to
        origin: Origin point for the clipping planes (default: (0, 0, 0))
        normals: List of normal vectors for the clipping planes (default: standard axes)
        
    Returns:
        The collection of clipping planes
        
    Raises:
        ValueError: If actor is invalid or parameters are invalid
    """
    if not isinstance(actor, (vtkActor, vtkVolume)):
        raise ValueError("Invalid actor provided")
        
    try:
        planes = vtk.vtkPlaneCollection()
        
        if origin is None:
            origin = (0, 0, 0)
            
        if normals is None:
            normals = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
            
        for normal in normals:
            plane = vtk.vtkPlane()
            plane.SetOrigin(*origin)
            plane.SetNormal(*normal)
            planes.AddItem(plane)
            
        if isinstance(actor, vtkActor):
            actor.GetMapper().SetClippingPlanes(planes)
        else:
            actor.GetMapper().SetClippingPlanes(planes)
            
        logger.debug("Added clipping planes to actor")
        return planes
    except Exception as e:
        logger.error(f"Failed to add clipping planes: {str(e)}")
        raise InteractionError(f"Failed to add clipping planes: {str(e)}")

def add_picking(
    interactor: vtk.vtkRenderWindowInteractor,
    renderer: vtk.vtkRenderer,
    callback: Optional[callable] = None
) -> vtk.vtkCellPicker:
    """
    Add picking functionality to the scene.
    
    Args:
        interactor: The VTK interactor to add picking to
        renderer: The VTK renderer to pick from
        callback: Optional callback function to handle pick events
        
    Returns:
        The cell picker instance
        
    Raises:
        ValueError: If interactor or renderer is invalid
    """
    if not isinstance(interactor, vtk.vtkRenderWindowInteractor):
        raise ValueError("Invalid interactor provided")
    if not isinstance(renderer, vtk.vtkRenderer):
        raise ValueError("Invalid renderer provided")
        
    try:
        picker = vtk.vtkCellPicker()
        picker.SetTolerance(1e-6)
        
        if callback:
            def on_pick(obj, event):
                picker.Pick(obj.GetEventPosition()[0], obj.GetEventPosition()[1], 0, renderer)
                callback(picker)
                
            interactor.AddObserver("LeftButtonPressEvent", on_pick)
            
        logger.debug("Added picking functionality")
        return picker
    except Exception as e:
        logger.error(f"Failed to add picking: {str(e)}")
        raise InteractionError(f"Failed to add picking: {str(e)}")

def add_measurement_tool(
    interactor: vtk.vtkRenderWindowInteractor,
    renderer: vtk.vtkRenderer
) -> vtk.vtkDistanceWidget:
    """
    Add a measurement tool to the scene.
    
    Args:
        interactor: The VTK interactor to add the measurement tool to
        renderer: The VTK renderer to measure in
        
    Returns:
        The distance widget instance
        
    Raises:
        ValueError: If interactor or renderer is invalid
    """
    if not isinstance(interactor, vtk.vtkRenderWindowInteractor):
        raise ValueError("Invalid interactor provided")
    if not isinstance(renderer, vtk.vtkRenderer):
        raise ValueError("Invalid renderer provided")
        
    try:
        widget = vtk.vtkDistanceWidget()
        widget.SetInteractor(interactor)
        widget.CreateDefaultRepresentation()
        widget.SetEnabled(True)
        
        logger.debug("Added measurement tool")
        return widget
    except Exception as e:
        logger.error(f"Failed to add measurement tool: {str(e)}")
        raise InteractionError(f"Failed to add measurement tool: {str(e)}")

def add_annotation_tool(
    interactor: vtk.vtkRenderWindowInteractor,
    renderer: vtk.vtkRenderer
) -> vtk.vtkTextWidget:
    """
    Add an annotation tool to the scene.
    
    Args:
        interactor: The VTK interactor to add the annotation tool to
        renderer: The VTK renderer to annotate
        
    Returns:
        The text widget instance
        
    Raises:
        ValueError: If interactor or renderer is invalid
    """
    if not isinstance(interactor, vtk.vtkRenderWindowInteractor):
        raise ValueError("Invalid interactor provided")
    if not isinstance(renderer, vtk.vtkRenderer):
        raise ValueError("Invalid renderer provided")
        
    try:
        widget = vtk.vtkTextWidget()
        widget.SetInteractor(interactor)
        widget.CreateDefaultRepresentation()
        widget.SetEnabled(True)
        
        logger.debug("Added annotation tool")
        return widget
    except Exception as e:
        logger.error(f"Failed to add annotation tool: {str(e)}")
        raise InteractionError(f"Failed to add annotation tool: {str(e)}")

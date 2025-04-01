"""
DICOM loader module for the Medical 3D Viewer application.
Provides functions for loading and processing DICOM image files.
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List
import vtk

# Configure logging
logger = logging.getLogger(__name__)

class DicomLoadError(Exception):
    """Custom exception for DICOM loading errors."""
    pass

def load_dicom(
    directory: str,
    window_width: float = 400,
    window_center: float = 40,
    color_table: Optional[List[Tuple[float, Tuple[float, float, float]]]] = None,
    opacity: float = 1.0,
    shade: bool = True,
    interpolation: str = "linear"
) -> vtk.vtkVolume:
    """
    Load DICOM files from a directory and return a VTK volume.
    
    Args:
        directory: Directory containing DICOM files
        window_width: Window width for intensity mapping (default: 400)
        window_center: Window center for intensity mapping (default: 40)
        color_table: List of (intensity, RGB) tuples for custom color mapping
        opacity: Opacity of the volume (0.0 to 1.0, default: 1.0)
        shade: Whether to enable shading (default: True)
        interpolation: Interpolation method ("linear" or "nearest", default: "linear")
        
    Returns:
        The VTK volume containing the loaded DICOM data
        
    Raises:
        DicomLoadError: If files cannot be loaded or parameters are invalid
    """
    if not os.path.exists(directory):
        raise DicomLoadError(f"Directory not found: {directory}")
        
    if not 0 <= opacity <= 1:
        raise ValueError("Opacity must be between 0 and 1")
        
    if interpolation not in ["linear", "nearest"]:
        raise ValueError("Invalid interpolation method")
        
    try:
        logger.info(f"Loading DICOM files from {directory}")
        
        # Create DICOM reader
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(directory)
        reader.Update()
        
        # Create volume mapper
        volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
        volume_mapper.SetInputConnection(reader.GetOutputPort())
        
        # Create volume property
        volume_property = vtk.vtkVolumeProperty()
        volume_property.SetShade(shade)
        
        if interpolation == "linear":
            volume_property.SetInterpolationTypeToLinear()
        else:
            volume_property.SetInterpolationTypeToNearest()
            
        # Set color transfer function
        color = vtk.vtkColorTransferFunction()
        if color_table:
            for intensity, rgb in color_table:
                color.AddRGBPoint(intensity, *rgb)
        else:
            # Default color mapping
            color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
            color.AddRGBPoint(window_center - window_width/2, 0.0, 0.0, 0.0)
            color.AddRGBPoint(window_center, 0.5, 0.5, 0.5)
            color.AddRGBPoint(window_center + window_width/2, 1.0, 1.0, 1.0)
            color.AddRGBPoint(1000, 1.0, 1.0, 1.0)
            
        volume_property.SetColor(color)
        
        # Set opacity transfer function
        opacity_tf = vtk.vtkPiecewiseFunction()
        opacity_tf.AddPoint(-1000, 0.0)
        opacity_tf.AddPoint(window_center - window_width/2, 0.0)
        opacity_tf.AddPoint(window_center, opacity)
        opacity_tf.AddPoint(window_center + window_width/2, opacity)
        opacity_tf.AddPoint(1000, opacity)
        volume_property.SetScalarOpacity(opacity_tf)
        
        # Create volume
        volume = vtk.vtkVolume()
        volume.SetMapper(volume_mapper)
        volume.SetProperty(volume_property)
        
        logger.debug("DICOM volume loaded successfully")
        return volume
        
    except Exception as e:
        logger.error(f"Failed to load DICOM: {str(e)}")
        raise DicomLoadError(f"Failed to load DICOM: {str(e)}")

def get_dicom_info(directory: str) -> Dict[str, Any]:
    """
    Get information about DICOM files in a directory.
    
    Args:
        directory: Directory containing DICOM files
        
    Returns:
        Dictionary containing DICOM information
        
    Raises:
        DicomLoadError: If files cannot be loaded
    """
    if not os.path.exists(directory):
        raise DicomLoadError(f"Directory not found: {directory}")
        
    try:
        # Create DICOM reader
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(directory)
        reader.Update()
        
        # Get image data
        image_data = reader.GetOutput()
        
        # Get dimensions and spacing
        dims = image_data.GetDimensions()
        spacing = image_data.GetSpacing()
        
        # Get scalar range
        scalar_range = image_data.GetScalarRange()
        
        # Get DICOM tags
        tags = {}
        for key in reader.GetMetaDataKeys():
            tags[key] = reader.GetMetaData(key)
            
        info = {
            "directory": directory,
            "num_files": len(os.listdir(directory)),
            "dimensions": dims,
            "spacing": spacing,
            "scalar_range": scalar_range,
            "dicom_tags": tags
        }
        
        logger.debug(f"Retrieved DICOM info: {info}")
        return info
        
    except Exception as e:
        logger.error(f"Failed to get DICOM info: {str(e)}")
        raise DicomLoadError(f"Failed to get DICOM info: {str(e)}")

def extract_slice(
    volume: vtk.vtkVolume,
    slice_index: int,
    orientation: str = "axial"
) -> vtk.vtkImageData:
    """
    Extract a 2D slice from a 3D volume.
    
    Args:
        volume: The VTK volume to extract from
        slice_index: Index of the slice to extract
        orientation: Slice orientation ("axial", "sagittal", or "coronal")
        
    Returns:
        The extracted 2D slice as vtkImageData
        
    Raises:
        ValueError: If volume is invalid or parameters are invalid
    """
    if not isinstance(volume, vtk.vtkVolume):
        raise ValueError("Invalid volume provided")
        
    if orientation not in ["axial", "sagittal", "coronal"]:
        raise ValueError("Invalid orientation")
        
    try:
        # Get volume mapper
        mapper = volume.GetMapper()
        input_data = mapper.GetInput()
        
        # Create extractor
        extractor = vtk.vtkExtractVOI()
        extractor.SetInputData(input_data)
        
        # Set slice range based on orientation
        dims = input_data.GetDimensions()
        if orientation == "axial":
            extractor.SetVOI(0, dims[0]-1, 0, dims[1]-1, slice_index, slice_index)
        elif orientation == "sagittal":
            extractor.SetVOI(slice_index, slice_index, 0, dims[1]-1, 0, dims[2]-1)
        else:  # coronal
            extractor.SetVOI(0, dims[0]-1, slice_index, slice_index, 0, dims[2]-1)
            
        extractor.Update()
        
        logger.debug(f"Extracted {orientation} slice at index {slice_index}")
        return extractor.GetOutput()
        
    except Exception as e:
        logger.error(f"Failed to extract slice: {str(e)}")
        raise DicomLoadError(f"Failed to extract slice: {str(e)}")

def create_mip(
    volume: vtk.vtkVolume,
    direction: Tuple[float, float, float] = (0, 0, 1)
) -> vtk.vtkImageData:
    """
    Create a maximum intensity projection (MIP) of a volume.
    
    Args:
        volume: The VTK volume to project
        direction: Projection direction vector (default: along z-axis)
        
    Returns:
        The MIP image as vtkImageData
        
    Raises:
        ValueError: If volume is invalid
    """
    if not isinstance(volume, vtk.vtkVolume):
        raise ValueError("Invalid volume provided")
        
    try:
        # Get volume mapper
        mapper = volume.GetMapper()
        input_data = mapper.GetInput()
        
        # Create MIP filter
        mip = vtk.vtkProjectedTexture()
        mip.SetInputData(input_data)
        mip.SetProjectionDirection(*direction)
        mip.Update()
        
        logger.debug("Created maximum intensity projection")
        return mip.GetOutput()
        
    except Exception as e:
        logger.error(f"Failed to create MIP: {str(e)}")
        raise DicomLoadError(f"Failed to create MIP: {str(e)}")

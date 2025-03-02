import vtk

def load_dicom(directory):
    """ Load DICOM files from a directory and return a VTK volume. """
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(directory)
    reader.Update()

    volume_mapper = vtk.vtkGPUVolumeRayCastMapper()
    volume_mapper.SetInputConnection(reader.GetOutputPort())

    volume_property = vtk.vtkVolumeProperty()
    volume_property.ShadeOn()
    volume_property.SetInterpolationTypeToLinear()

    color = vtk.vtkColorTransferFunction()
    color.AddRGBPoint(-1000, 0.0, 0.0, 0.0)
    color.AddRGBPoint(400, 1.0, 1.0, 1.0)
    volume_property.SetColor(color)

    volume = vtk.vtkVolume()
    volume.SetMapper(volume_mapper)
    volume.SetProperty(volume_property)

    return volume

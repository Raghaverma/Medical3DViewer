import vtk

def load_model(file_path):
    """ Load STL or OBJ files and return a VTK actor. """
    if file_path.endswith(".stl"):
        reader = vtk.vtkSTLReader()
    elif file_path.endswith(".obj"):
        reader = vtk.vtkOBJReader()
    else:
        raise ValueError("Unsupported file format")

    reader.SetFileName(file_path)
    reader.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToPhong()

    return actor

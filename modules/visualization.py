import vtk

def add_axes(interactor):
    """ Add an orientation axes widget to the scene. """
    axes = vtk.vtkAxesActor()
    widget = vtk.vtkOrientationMarkerWidget()
    widget.SetOutlineColor(1, 1, 1)
    widget.SetOrientationMarker(axes)
    widget.SetInteractor(interactor)
    widget.SetViewport(0.0, 0.0, 0.2, 0.2)
    widget.SetEnabled(True)
    widget.InteractiveOff()

def add_bounding_box(renderer):
    """ Add a bounding box to the scene. """
    cube_axes = vtk.vtkCubeAxesActor()
    cube_axes.SetCamera(renderer.GetActiveCamera())
    cube_axes.DrawXGridlinesOn()
    cube_axes.DrawYGridlinesOn()
    cube_axes.DrawZGridlinesOn()
    renderer.AddActor(cube_axes)

def add_lighting(renderer):
    """ Add a light source for better visualization. """
    light = vtk.vtkLight()
    light.SetFocalPoint(0, 0, 0)
    light.SetPosition(100, 100, 100)
    renderer.AddLight(light)

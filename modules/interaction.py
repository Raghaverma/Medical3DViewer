import vtk

def set_interaction_style(interactor):
    """ Set trackball camera interaction style. """
    style = vtk.vtkInteractorStyleTrackballCamera()
    interactor.SetInteractorStyle(style)

def add_clipping_planes(actor):
    """ Add slicing planes for better visualization. """
    planes = vtk.vtkPlaneCollection()
    
    for i in range(3):
        plane = vtk.vtkPlane()
        plane.SetOrigin(0, 0, 0)
        plane.SetNormal(1 if i == 0 else 0, 1 if i == 1 else 0, 1 if i == 2 else 0)
        planes.AddItem(plane)

    actor.GetMapper().SetClippingPlanes(planes)

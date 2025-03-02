import vtk

def create_annotation(point):
    """ Create a text annotation at a specific point. """
    text_actor = vtk.vtkTextActor()
    text_actor.SetTextScaleModeToNone()
    text_actor.SetInput(f"({point[0]:.2f}, {point[1]:.2f}, {point[2]:.2f})")
    text_actor.GetTextProperty().SetFontSize(20)
    return text_actor

def measure_distance(p1, p2):
    """ Measure distance between two points. """
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2) ** 0.5

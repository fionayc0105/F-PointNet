import os
import numpy
import vtk
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
SUNRGBD_ROOT = os.path.join(BASE_DIR, '../sunrgbd_data/matlab/SUNRGBDtoolbox/mysunrgbd')
SUNRGBD_DIR = os.path.join(SUNRGBD_ROOT, 'training')


def flip_axis_to_camera(pc):
    pc2 = np.copy(pc)
    pc2[:, [0, 1, 2]] = pc2[:, [0, 2, 1]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[:, 1] *= -1
    return pc2


def flip_axis_x(pc): # cam X,Y,Z = depth X,-Z,Y
    pc2 = np.copy(pc)
    pc2[:, [0, 1, 2]] = pc2[:, [0, 1, 2]]  # cam X,Y,Z = depth X,-Z,Y
    pc2[:, 0] *= -1
    return pc2


def display(source_data, window_name):
    pc = source_data[:, 0:3]
    # 新建 vtkPoints 实例
    points = vtk.vtkPoints()
    pointColors = vtk.vtkUnsignedCharArray()
    pointColors.SetNumberOfComponents(3)
    pc = flip_axis_x(pc)

    for i in range(len(pc)):
        # option1: 一次性導入
        # points.SetData(numpy_to_vtk(pc))
        # option2: 單點設定
        points.InsertNextPoint(pc[i][0], pc[i][1], pc[i][2])
        color = [int(source_data[i, 3] * 255), int(source_data[i, 4] * 255), int(source_data[i, 5] * 255)]
        pointColors.InsertNextTypedTuple(color)

    # 新建vtkPolyData實例
    polydata = vtk.vtkPolyData()
    # 設置點座標
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(pointColors)

    # 頂點相關的filter
    vertex = vtk.vtkVertexGlyphFilter()
    vertex.SetInputData(polydata)

    # mapper實例
    mapper = vtk.vtkPolyDataMapper()
    # 關聯filter輸出
    mapper.SetInputConnection(vertex.GetOutputPort())
    # actor 實例
    actor = vtk.vtkActor()
    # 關聯mapper
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1, 0, 0) # R,G,B
    # Create a render window
    render = vtk.vtkRenderer()
    # Insert Actor
    render.AddActor(actor)
    render.SetBackground(0, 0, 0)
    # Renderer Window
    renderWindows = vtk.vtkRenderWindow()
    renderWindows.AddRenderer(render)
    renderWindows.SetSize(1200, 1200)
    renderWindows.SetWindowName(window_name)

    # 加上3軸座標軸
    axis_actor = vtk.vtkAxesActor()
    axis_actor.SetScale(10)
    render.AddActor(axis_actor)
    # System Event
    iwin_render = vtk.vtkRenderWindowInteractor()
    iwin_render.SetRenderWindow(renderWindows)
    # Style
    iwin_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())
    iwin_render.Initialize()
    #renderWindows.Render()
    iwin_render.Start()


if __name__ == '__main__':
    filename = os.path.join(SUNRGBD_DIR, 'depth/012004.txt')
    source_data = numpy.loadtxt(filename)
    # pc = flip_axis_to_camera(source_data)
    display(source_data, "pc color")

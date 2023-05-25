import os
import numpy
import vtk
from vtk.util.numpy_support import numpy_to_vtk

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
SUNRGBD_ROOT = os.path.join(BASE_DIR, '../sunrgbd_data/matlab/SUNRGBDtoolbox/mysunrgbd')
SUNRGBD_DIR = os.path.join(SUNRGBD_ROOT, 'training')

def display(source_data, window_name):
    # 读取 txt 文档
    # filename = os.path.join(SUNRGBD_DIR, 'depth/009986.txt')
    # source_data = numpy.loadtxt(filename)
    pc = source_data[:, 0:3]
    # 新建 vtkPoints 实例
    points = vtk.vtkPoints()
    pointColors = vtk.vtkUnsignedCharArray()
    pointColors.SetNumberOfComponents(3)

    for i in range(len(pc)):
        # option1: 一次性導入
        # points.SetData(numpy_to_vtk(pc))
        # option2: 單點設定
        points.InsertNextPoint(pc[i][0], pc[i][1], pc[i][2])
        # color = [int(source_data[i, 5]*255), int(source_data[i, 4]*255), int(source_data[i, 3]*255)]
        color = [int(source_data[i, 3] * 255), int(source_data[i, 4] * 255), int(source_data[i, 5] * 255)]
        pointColors.InsertNextTypedTuple(color)


    # 新建 vtkPolyData 实例
    polydata = vtk.vtkPolyData()
    # 设置点坐标
    polydata.SetPoints(points)
    polydata.GetPointData().SetScalars(pointColors)

    # 顶点相关的 filter
    vertex = vtk.vtkVertexGlyphFilter()
    vertex.SetInputData(polydata)

    # mapper 实例
    mapper = vtk.vtkPolyDataMapper()
    # 关联 filter 输出
    mapper.SetInputConnection(vertex.GetOutputPort())

    # actor 实例
    actor = vtk.vtkActor()
    # 关联 mapper
    actor.SetMapper(mapper)
    # 红色点显示
    actor.GetProperty().SetColor(1, 0, 0) # R,G,B

    # Create a render window
    render = vtk.vtkRenderer()
    #colors = vtk.vtkNamedColors()
    # Insert Actor
    render.AddActor(actor)
    render.SetBackground(0, 0, 0)
    #render.SetBackground(colors.GetColor3d('RosyBrown'))
    # Renderer Window
    renderWindows = vtk.vtkRenderWindow()
    renderWindows.AddRenderer(render)
    renderWindows.SetSize(1200, 1200)
    renderWindows.SetWindowName(window_name)

    # System Event
    iwin_render = vtk.vtkRenderWindowInteractor()
    iwin_render.SetRenderWindow(renderWindows)
    # Style
    iwin_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())

    iwin_render.Initialize()
    #renderWindows.Render()
    iwin_render.Start()

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


def project_pc_to_image(pc):
    K = np.eye(3)
    K[0, 0] = 1081.372
    K[1, 1] = 1081.372
    K[0, 2] = 959.5
    K[1, 2] = 539.5
    uv = np.dot(pc[:, 0:3], np.transpose(K)) # (n,3)
    uv[:,0] /= uv[:,2]
    uv[:,1] /= uv[:,2]
    return uv[:,0:2], pc[:,2]


def find_roi_pc(pc, xmin, xmax, ymin, ymax):
    pc_image_coord, _  = project_pc_to_image(pc)
    box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & (
            pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)

    pc_in_box_fov = pc[box_fov_inds, :]
    display(pc_in_box_fov, "test pc")


def display(source_data, window_name):
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
    pc = flip_axis_x(source_data)
    display(pc, "pc color")
    # 找出落在某個影像範圍內的點雲
    x = 695
    y = 210
    w = 418
    h = 363
    find_roi_pc(pc, x, x+w, y, y+h)
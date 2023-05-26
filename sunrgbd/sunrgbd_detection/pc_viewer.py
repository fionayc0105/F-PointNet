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
    # 新建vtkPoints實例
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
    # Create a renderer window
    renderer = vtk.vtkRenderer()
    # Insert Actor
    renderer.AddActor(actor)
    renderer.SetBackground(0, 0, 0)
    # Renderer Window
    renderWnd = vtk.vtkRenderWindow()
    renderWnd.AddRenderer(renderer)
    renderWnd.SetSize(1200, 1200)
    renderWnd.SetWindowName(window_name)
    # 加上3軸座標軸
    axis_actor = vtk.vtkAxesActor()
    axis_actor.SetScale(5)
    renderer.AddActor(axis_actor)
    # 畫出3d box
    box3d = np.array([[0, 0, 3], [0, 0, 0], [2, 0, 0], [2, 0, 3], [0, 1, 3], [0, 1, 0], [2, 1, 0], [2, 1, 3]], dtype = float)
    plot_3d_box(renderer, box3d)

    # System Event
    iwin_render = vtk.vtkRenderWindowInteractor()
    iwin_render.SetRenderWindow(renderWnd)
    # Style
    iwin_render.SetInteractorStyle(vtk.vtkInteractorStyleMultiTouchCamera())
    iwin_render.Initialize()
    iwin_render.Start()
    return renderWnd


# 標記出3d box的角點
def plot_3d_box(renderer, box3d):
    # 創建vtkPoints對象，存儲bounding box的頂點座標
    points = vtk.vtkPoints()
    for i in range(len(box3d)):
        x, y, z = box3d[i]
        points.InsertNextPoint(x, y, z);

    # 創建vtkCellArray對象，定義bounding box的邊界
    edges = vtk.vtkCellArray()
    for k in range(0, 4):
       for q in range(3):
           if q == 0:
               i, j = k, (k + 1) % 4;
           elif q== 1:
               i, j = k + 4, (k + 1) % 4 + 4;
           else:
                i, j = k, k + 4;
           edges.InsertNextCell(2)
           edges.InsertCellPoint(i)
           edges.InsertCellPoint(j)

    # 創建vtkPolyData對象，將點和邊結合起來
    polyData = vtk.vtkPolyData()
    polyData.SetPoints(points)
    polyData.SetLines(edges)

    # 創建vtkPolyDataMapper對象，將polyData映射到渲染器上
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polyData)

    # 創建vtkActor對象，將mapper添加到actor上
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 1.0, 0)  # 設定線的顏色
    # 創建vtkRenderer對象和vtkRenderWindow對象，並將actor添加到渲染器中
    renderer.AddActor(actor)
    return renderer






    # # 創建vtkPoints對象，存儲點的坐標
    # points = vtk.vtkPoints()
    # points.InsertNextPoint(1, 0, 0)  # 第一個點的坐標
    # points.InsertNextPoint(5, 5, 5)  # 第二個點的坐標
    # # 創建vtkCellArray對象，存儲線的索引
    # lines = vtk.vtkCellArray()
    # line = vtk.vtkLine()
    # line.GetPointIds().SetId(0, 0)  # 第一條線的第一個點的索引
    # line.GetPointIds().SetId(1, 1)  # 第一條線的第二個點的索引
    # lines.InsertNextCell(line)
    # # 創建vtkPolyData對象，將點和線結合起來
    # polyData = vtk.vtkPolyData()
    # polyData.SetPoints(points)
    # polyData.SetLines(lines)
    # # 創建vtkPolyDataMapper對象，將polyData映射到渲染器上
    # mapper = vtk.vtkPolyDataMapper()
    # mapper.SetInputData(polyData)
    # # 創建vtkActor對象，將mapper添加到actor上
    # actor = vtk.vtkActor()
    # actor.SetMapper(mapper)
    # actor.GetProperty().SetColor(1.0, 0, 1.0) # 設定線的顏色
    # render.AddActor(actor)
    # return render


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
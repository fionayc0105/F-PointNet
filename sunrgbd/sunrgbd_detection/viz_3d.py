import pickle
import numpy as np
import argparse
from PIL import Image
import cv2
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

import roi_seg_box3d_dataset
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
from sunrgbd_data import sunrgbd_object
from utils import load_zipped_pickle, compute_box_3d, draw_projected_box3d, draw_label
sys.path.append(os.path.join(BASE_DIR, '../../train'))
sys.path.append(os.path.join(BASE_DIR, '../../mayavi'))
from box_util import box3d_iou

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', default=None, help='data path for .pickle file, the one used for val in train.py [default: None]')
parser.add_argument('--result_path', default=None, help='result path for .pickle file from test.py [default: None]')
parser.add_argument('--viz', action='store_true', help='to visualize error result.')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from data file from rgb detection.')
FLAGS = parser.parse_args()

IMG_DIR = '/home/fiona/ws/PointNet/frustum-pointnets/src/sunrgbd/sunrgbd_data/matlab/SUNRGBDtoolbox/mysunrgbd/training/image'
SAVE_DIR = '/home/fiona/ws/PointNet/frustum-pointnets/src/sunrgbd/sunrgbd_detection/image_results'
TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(npoints=2048, split='val', rotate_to_center=True, overwritten_data_path=FLAGS.data_path, from_rgb_detection=FLAGS.from_rgb_detection)
dataset = sunrgbd_object('/home/fiona/ws/PointNet/frustum-pointnets/src/sunrgbd/sunrgbd_data/matlab/SUNRGBDtoolbox/mysunrgbd', 'training')
VISU = FLAGS.viz
ps_list, segp_list, center_list, heading_cls_list, heading_res_list, size_cls_list, size_res_list, rot_angle_list, score_list = load_zipped_pickle(FLAGS.result_path)

total_cnt = 0
correct_cnt = 0
type_whitelist=['bed','table','sofa','chair','toilet','desk','dresser','night_stand','bookshelf','bathtub']
class_correct_cnt = {classname:0 for classname in type_whitelist}
class_total_cnt = {classname:0 for classname in type_whitelist}
val_idx = [line.rstrip() for line in open(os.path.join(ROOT_DIR, 'sunrgbd_data/matlab/SUNRGBDtoolbox/mysunrgbd/training/val_data_idx.txt'))]

#################
# drawing color
#################
COLORS = { 'bed': (226, 126, 37),
           'table': (91, 91, 255),
           'sofa': (255, 192, 30),
           'chair': (25, 163, 255),
           'toilet': (0, 0, 255),
           'desk': (182, 188, 8),
           'dresser': (249, 148, 165),
           'night_stand': (61, 231, 98),
           'bookshelf': (64, 128, 0),
           'bathtub': (64, 0, 64)
           }



for i in range(len(val_idx)):
    img_id = int(str(val_idx[i]))
    calib = dataset.get_calibration(img_id)
    img_filename = os.path.join(IMG_DIR, '%06d.jpg' % (img_id))
    img2 = cv2.imread(img_filename)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    objects = dataset.get_label_objects(img_id)

    for obj in objects:
        if obj.classname not in ['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand',
                                 'bookshelf', 'bathtub']: continue
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib)
        img2 = draw_projected_box3d(img2, box3d_pts_2d, color = COLORS[obj.classname])
        img2 = draw_label(img2, box3d_pts_2d[0,0] ,box3d_pts_2d[0,1], obj.classname,  COLORS[obj.classname])
        print(obj.classname)
    print("-----------------------------\n")
    # Image.fromarray(img2).show()
    output_filename = os.path.join(SAVE_DIR, '%06d.jpg' % (img_id))
    cv2.imwrite(output_filename, img2)

    # input()
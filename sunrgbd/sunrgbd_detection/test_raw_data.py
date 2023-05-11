''' Testing Frustum PointNets on SUN-RGBD dataset.

Author: Charles R. Qi
Date: October 2017
'''

import argparse
import importlib
import os
import sys
import numpy as np
import tensorflow as tf
import pandas as pd
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)  # model
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, '../sunrgbd_data'))
sys.path.append(os.path.join(BASE_DIR, '../../kitti'))

import roi_seg_box3d_dataset
from roi_seg_box3d_dataset import NUM_CLASS, NUM_SIZE_CLUSTER, NUM_HEADING_BIN
from utils import load_zipped_pickle, draw_projected_box3d, draw_label, compute_box_3d, rotz, roty, rotx
from roi_seg_box3d_dataset import class2angle, class2size, get_3d_box
import sys
import os
from sunrgbd_data import sunrgbd_object

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--model', default='model', help='Model name [default: model]')
parser.add_argument('--model_path', default='log/model.ckpt',
                    help='model checkpoint file path [default: log/model.ckpt]')
parser.add_argument('--output', default='test_results', help='output filename [default: test_results]')
parser.add_argument('--data_path', default=None, help='data path [default: None]')
parser.add_argument('--from_rgb_detection', action='store_true', help='test from data file from rgb detection.')
parser.add_argument('--idx_path', default=None,
                    help='filename of txt where each line is a data idx, used for rgb detection -- write <id>.txt for all frames. [default: None]')
parser.add_argument('--dump_result', action='store_true', help='dump result to .pickle file')
FLAGS = parser.parse_args()

MODEL_PATH = FLAGS.model_path
GPU_INDEX = FLAGS.gpu
NUM_POINT = FLAGS.num_point
MODEL = importlib.import_module(FLAGS.model)  # import network module
NUM_CHANNEL = 6

SAVE_DIR = os.path.join(BASE_DIR, 'viz_results')
SUNRGBD_ROOT = os.path.join(BASE_DIR, '../sunrgbd_data/matlab/SUNRGBDtoolbox/mysunrgbd')
SUNRGBD_DIR = os.path.join(SUNRGBD_ROOT, 'training')
IMG_DIR = os.path.join(SUNRGBD_DIR, 'image')
TEST_DATASET = roi_seg_box3d_dataset.ROISegBoxDataset(npoints=2048, split='val', rotate_to_center=True, overwritten_data_path=FLAGS.data_path, from_rgb_detection=FLAGS.from_rgb_detection, one_hot=True)

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


def get_model(batch_size, num_point):
    with tf.Graph().as_default():
        with tf.device('/gpu:' + str(GPU_INDEX)):
            pointclouds_pl, one_hot_vec_pl, labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl, size_class_label_pl, size_residual_label_pl = MODEL.placeholder_inputs(
                batch_size, num_point)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            pred, end_points = MODEL.get_model(pointclouds_pl, one_hot_vec_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl, centers_pl, heading_class_label_pl, heading_residual_label_pl,
                                  size_class_label_pl, size_residual_label_pl, end_points)
            saver = tf.train.Saver()
        # for v in tf.global_variables():
        #    print(v.name)
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        sess = tf.Session(config=config)
        # Restore variables from disk.
        saver.restore(sess, MODEL_PATH)
        ops = {'pointclouds_pl': pointclouds_pl,
               'one_hot_vec_pl': one_hot_vec_pl,
               'labels_pl': labels_pl,
               'centers_pl': centers_pl,
               'heading_class_label_pl': heading_class_label_pl,
               'heading_residual_label_pl': heading_residual_label_pl,
               'size_class_label_pl': size_class_label_pl,
               'size_residual_label_pl': size_residual_label_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'center': end_points['center'],
               'end_points': end_points,
               'loss': loss}
        return sess, ops


def softmax(x):
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape) - 1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape) - 1, keepdims=True)
    return probs


def inference(sess, ops, pc, one_hot_vec, batch_size):
    ''' pc: BxNx3 array, Bx3 array, return BxN pred and Bx3 centers '''
    assert pc.shape[0] % batch_size == 0
    num_batches = int(pc.shape[0] / batch_size)
    logits = np.zeros((pc.shape[0], pc.shape[1], 2))
    centers = np.zeros((pc.shape[0], 3))
    heading_logits = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    heading_residuals = np.zeros((pc.shape[0], NUM_HEADING_BIN))
    size_logits = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER))
    size_residuals = np.zeros((pc.shape[0], NUM_SIZE_CLUSTER, 3))
    scores = np.zeros((pc.shape[
                           0],))  # score that indicates confidence in 3d box prediction (mask logits+heading+size); no confidence for the center...

    ep = ops['end_points']
    for i in range(num_batches):
        feed_dict = {ops['pointclouds_pl']: pc[i * batch_size:(i + 1) * batch_size, ...],
                     ops['one_hot_vec_pl']: one_hot_vec[i * batch_size:(i + 1) * batch_size, :],
                     ops['is_training_pl']: False}
        batch_logits, batch_centers, batch_heading_scores, batch_heading_residuals, batch_size_scores, batch_size_residuals = sess.run(
            [ops['pred'], ops['center'], ep['heading_scores'], ep['heading_residuals'], ep['size_scores'],
             ep['size_residuals']], feed_dict=feed_dict)
        logits[i * batch_size:(i + 1) * batch_size, ...] = batch_logits
        centers[i * batch_size:(i + 1) * batch_size, ...] = batch_centers
        heading_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_scores
        heading_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_heading_residuals
        size_logits[i * batch_size:(i + 1) * batch_size, ...] = batch_size_scores
        size_residuals[i * batch_size:(i + 1) * batch_size, ...] = batch_size_residuals

        # Compute scores
        batch_seg_prob = softmax(batch_logits)[:, :, 1]  # BxN
        batch_seg_mask = np.argmax(batch_logits, 2)  # BxN
        mask_mean_prob = np.sum(batch_seg_prob * batch_seg_mask, 1)  # B,
        mask_mean_prob = mask_mean_prob / np.sum(batch_seg_mask, 1)  # B,
        heading_prob = np.max(softmax(batch_heading_scores), 1)  # B
        size_prob = np.max(softmax(batch_size_scores), 1)  # B,
        batch_scores = np.log(mask_mean_prob) + np.log(heading_prob) + np.log(size_prob)
        scores[i * batch_size:(i + 1) * batch_size] = batch_scores
        # Finished computing scores

    heading_cls = np.argmax(heading_logits, 1)  # B
    size_cls = np.argmax(size_logits, 1)  # B
    return np.argmax(logits, 2), centers, heading_cls, np.array(
        [heading_residuals[i, heading_cls[i]] for i in range(pc.shape[0])]), size_cls, np.vstack(
        [size_residuals[i, size_cls[i], :] for i in range(pc.shape[0])]), scores


def get_center_view_rot_angle(frustum_angle):
    return np.pi / 2.0 + frustum_angle


def rotate_pc_along_y(pc, rot_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    cosval = np.cos(rot_angle)
    sinval = np.sin(rot_angle)
    rotmat = np.array([[cosval, -sinval], [sinval, cosval]])
    pc[:, [0, 2]] = np.dot(pc[:, [0, 2]], np.transpose(rotmat))
    return pc


def get_center_view_point_set(pc, frustum_angle):
    ''' Input ps is NxC points with first 3 channels as XYZ
        z is facing forward, x is left ward, y is downward
    '''
    point_set = np.copy(pc)
    return rotate_pc_along_y(point_set, get_center_view_rot_angle(frustum_angle))


def get_frustum_angle(box2d, calib):
    xmin, ymin, xmax, ymax = box2d
    box2d_center = np.array([(xmin + xmax) / 2.0, (ymin + ymax) / 2.0])
    uvdepth = np.zeros((1, 3))
    uvdepth[0, 0:2] = box2d_center
    uvdepth[0, 2] = 20  # some random depth
    box2d_center_upright_camera = calib.project_image_to_upright_camerea(uvdepth)
    frustum_angle = -1 * np.arctan2(box2d_center_upright_camera[0, 2], box2d_center_upright_camera[
        0, 0])  # angle as to positive x-axis as in the Zoox paper
    return frustum_angle


def get_batch(data_idx, type_whitelist=['bed', 'table', 'sofa', 'chair', 'toilet', 'desk', 'dresser', 'night_stand',
                                          'bookshelf', 'bathtub']):
    dataset = sunrgbd_object(SUNRGBD_ROOT, 'training')
    pc_list = []  # point cloud list,  channel number = 6, xyz,rgb in upright depth coord
    onehot_list = []  # angle of 2d box center from pos x-axis (clockwise)
    rot_angle_list = []
    type_list = []
    box2d_list = []

    calib = dataset.get_calibration(data_idx)
    objects = dataset.get_label_objects(data_idx)
    pc_upright_depth = dataset.get_depth(data_idx)
    pc_upright_camera = np.zeros_like(pc_upright_depth)
    pc_upright_camera[:, 0:3] = calib.project_upright_depth_to_upright_camera(pc_upright_depth[:, 0:3])
    pc_upright_camera[:, 3:] = pc_upright_depth[:, 3:]
    img = dataset.get_image(data_idx)
    img_height, img_width, img_channel = img.shape
    pc_image_coord, _ = calib.project_upright_depth_to_image(pc_upright_depth)

    for obj_idx in range(len(objects)):
        obj = objects[obj_idx]
        if obj.classname not in type_whitelist: continue
        box2d = obj.box2d
        xmin, ymin, xmax, ymax = box2d

        box_fov_inds = (pc_image_coord[:, 0] < xmax) & (pc_image_coord[:, 0] >= xmin) & (
                pc_image_coord[:, 1] < ymax) & (pc_image_coord[:, 1] >= ymin)
        # step1: get point cloud
        pc_in_box_fov = pc_upright_camera[box_fov_inds, :]

        # step2: calculate the frustum angle and rotation angle
        frustum_angle = get_frustum_angle(box2d, calib)
        rot_angle = get_center_view_rot_angle(frustum_angle)

        # step3: sub-sample
        num_point = pc_in_box_fov.shape[0]
        if num_point > 2048:
            choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace=False)
            pc_in_box_fov = pc_in_box_fov[choice, :]

        # step4: rotate to center
        pc_in_box_fov = get_center_view_point_set(pc_in_box_fov, frustum_angle)

        # step5: Resample
        choice = np.random.choice(pc_in_box_fov.shape[0], 2048, replace=True)
        pc_in_box_fov = pc_in_box_fov[choice, :]
        pc_list.append(pc_in_box_fov)

        # one hot vector
        data_dum = pd.get_dummies(type_whitelist)
        pd.DataFrame(data_dum)
        one_hot_vec = data_dum["table"].tolist()
        onehot_list.append(one_hot_vec)
        rot_angle_list.append(rot_angle)
        type_list.append(obj.classname)
        box2d_list.append(box2d)

    return pc_list, onehot_list, rot_angle_list, type_list, box2d_list,  calib


def test_data(viz=True):
    batch_size = 1
    num_batches = 3
    cur_batch_size = 1
    batch_data_to_feed = np.zeros((batch_size, NUM_POINT, NUM_CHANNEL))
    batch_one_hot_to_feed = np.zeros((batch_size, NUM_CLASS))
    sess, ops = get_model(batch_size=batch_size, num_point=NUM_POINT)
    filename_val = os.path.join(SUNRGBD_DIR, "val_data_idx.txt")
    val_list = [int(line.rstrip()) for line in open(filename_val)]

    for j in range(len(val_list)):
        batch_idx = val_list[j]
        print ("----------------------------------------------\n")
        print ("Image:", batch_idx)
        filename = os.path.join(SAVE_DIR, '%06d.jpg' % (batch_idx))
        if os.path.isfile(filename):
            img = cv2.imread(filename)
        else:
            img_filename = os.path.join(IMG_DIR, '%06d.jpg' % (batch_idx))
            img = cv2.imread(img_filename)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pc_list, onehot_list, rot_angle_list, type_list, box2d_list,  calib = get_batch(batch_idx)
        for i in range(len(pc_list)):
            batch_data = pc_list[i]
            one_hot_vec = onehot_list[i]
            rot_angle = rot_angle_list[i]
            box2d = box2d_list[i]
            batch_data_to_feed[0:cur_batch_size, ...] = batch_data
            batch_one_hot_to_feed[0:cur_batch_size, :] = one_hot_vec
            batch_output, batch_center_pred, batch_hclass_pred, batch_hres_pred, batch_sclass_pred, batch_sres_pred, batch_scores = inference(
                sess, ops, batch_data_to_feed, batch_one_hot_to_feed, batch_size=batch_size)
            # Option 1:
            heading_angle = class2angle(batch_hclass_pred[0], batch_hres_pred[0], NUM_HEADING_BIN) + rot_angle
            box_size = class2size(batch_sclass_pred[0], batch_sres_pred[0])  # l,w,h
            center= rotate_pc_along_y(np.expand_dims(batch_center_pred[0], 0), -rot_angle).squeeze()
            corners_3d = get_3d_box(box_size, heading_angle, center)
            corners_upright_depth = calib.project_upright_camera_to_upright_depth(corners_3d)
            box3d_pts_2d, _ = calib.project_upright_depth_to_image(corners_upright_depth)

            # # Option 2:
            # heading_angle = class2angle(batch_hclass_pred[0], batch_hres_pred[0], NUM_HEADING_BIN)
            # box_size = class2size(batch_sclass_pred[0], batch_sres_pred[0])  # l,w,h
            # center = batch_center_pred[0]
            # corners_3d = get_3d_box(box_size, heading_angle, center)
            # corners_3d = rotate_pc_along_y(corners_3d, -rot_angle)
            # center2 = [np.average(corners_3d[:, 0]), np.average(corners_3d[:, 1]), np.average(corners_3d[:,2])]
            # corners_upright_depth = calib.project_upright_camera_to_upright_depth(corners_3d)
            # box3d_pts_2d, _ = calib.project_upright_depth_to_image(corners_upright_depth)

            img = draw_projected_box3d(img, box3d_pts_2d, color=COLORS[type_list[i]])
            img = draw_label(img, box3d_pts_2d[0, 0], box3d_pts_2d[0, 1], type_list[i], COLORS[type_list[i]])
        cv2.imwrite(filename, img)


if __name__ == '__main__':
    test_data()

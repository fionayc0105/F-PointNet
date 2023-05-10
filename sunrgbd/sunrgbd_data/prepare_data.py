import os
import sys
from sklearn import model_selection
from sunrgbd_data import sunrgbd_object

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUNRGBD_ROOT = os.path.join(BASE_DIR, 'matlab/SUNRGBDtoolbox/mysunrgbd')
SUNRGBD_DIR = os.path.join(SUNRGBD_ROOT, 'training')
SUNRGBD_DARASET = sunrgbd_object(SUNRGBD_ROOT, 'training')
def save(filename, contents):
    f = open(filename, 'w')
    f.writelines(contents)
    f.close()


# 將數據集依照比例分割為 train/valid/test
def split_data(idx_list, ratio_train, ratio_valid, ratio_test):
    train, test = model_selection.train_test_split(idx_list, test_size=ratio_test)
    ratio = ratio_valid/(1-ratio_test)
    train, valid = model_selection.train_test_split(train, test_size=ratio)
    train.sort()
    valid.sort()
    test.sort()

    c_train = ''.join(['%s' % (x) + "\n" for x in train])
    c_valid = ''.join(['%s' % (x) + "\n" for x in valid])
    filename_train = os.path.join(SUNRGBD_DIR, "train_data_idx.txt")
    filename_valid = os.path.join(SUNRGBD_DIR, "val_data_idx.txt")
    save(filename_train, c_train)
    save(filename_valid, c_valid)


# 取得所有包含某類別的數據label以及data_idx
def categorize_data(idx_list, classname):
    idx_name = os.path.join(SUNRGBD_DIR, "data_idx_%s.txt" % (classname))
    idx_fp = open(idx_name, "a+")
    for i in range(len(idx_list)):
        data_idx = idx_list[i]
        objects = SUNRGBD_DARASET.get_label_objects(data_idx)
        label_name = os.path.join(SUNRGBD_DIR, "label_dimension/%06d.txt" % (data_idx))
        relabel_dir = os.path.join(SUNRGBD_DIR, "label_dimension_%s" % (classname))
        if not os.path.exists(relabel_dir):
            os.makedirs(relabel_dir)
        relabel_name = os.path.join(relabel_dir, "%06d.txt" % (data_idx))
        lines = [line.rstrip() for line in open(label_name)]
        is_head = True
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if obj.classname == classname:
                fp = open(relabel_name, "a+")
                if is_head:
                    idx_fp.write("%06d\n" % (data_idx))
                    is_head = False
                else:
                    fp.write('\n')
                fp.write(lines[obj_idx])
                fp.close()
    idx_fp.close()


if __name__=='__main__':
    idx_list = list(range(1, 1+10335))
    # split_data(idx_list, 0.8, 0.19, 0.01)

    # 分離出單一類別的數據進行訓練
    categorize_data(idx_list, 'chair')


import os
import sys
from sklearn import model_selection
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
SAVE_DIR = os.path.join(BASE_DIR, 'matlab/SUNRGBDtoolbox/mysunrgbd/training')

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
    filename_train = os.path.join(SAVE_DIR, "train_data_idx.txt")
    filename_valid = os.path.join(SAVE_DIR, "val_data_idx.txt")
    save(filename_train, c_train)
    save(filename_valid, c_valid)


if __name__=='__main__':
    idx_list = list(range(1, 1+10335))
    split_data(idx_list, 0.8, 0.19, 0.01)


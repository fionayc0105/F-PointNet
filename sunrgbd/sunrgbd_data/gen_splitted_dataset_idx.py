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


def split_data(start_idx, count, ratio_train, ratio_valid, ratio_test):
    df = list(range(start_idx, start_idx + count))

    train, test = model_selection.train_test_split(df, test_size=ratio_test)
    ratio = ratio_valid/(1-ratio_test)
    train, valid = model_selection.train_test_split(train, test_size=ratio)
    train.sort()
    valid.sort()
    test.sort()

    c_train = ''.join(['%06d' % (x) + "\n" for x in train])
    c_valid = ''.join(['%06d' % (x) + "\n" for x in valid])
    filename_train = os.path.join(SAVE_DIR, "train_data_idx.txt")
    filename_valid = os.path.join(SAVE_DIR, "val_data_idx.txt")
    save(filename_train, c_train)
    save(filename_valid, c_valid)

if __name__=='__main__':
    split_data(1, 10335, 0.8, 0.19, 0.01)

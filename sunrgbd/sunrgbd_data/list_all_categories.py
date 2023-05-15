import os

import numpy as np

from sunrgbd_data import sunrgbd_object

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SUNRGBD_ROOT = os.path.join(BASE_DIR, 'matlab/SUNRGBDtoolbox/mysunrgbd')
SUNRGBD_DIR = os.path.join(SUNRGBD_ROOT, 'training')
SUNRGBD_DARASET = sunrgbd_object(SUNRGBD_ROOT, 'training')


def save(filename, contents):
    f = open(filename, 'w')
    f.writelines(contents)
    f.close()


def list_all():
    cate_list = []
    counts = {}
    for data_idx in range(1, 10335):
        objects = SUNRGBD_DARASET.get_label_objects(data_idx)
        for obj_idx in range(len(objects)):
            obj = objects[obj_idx]
            if counts.get(obj.classname) == None:
                counts[obj.classname] = 1
            else:
                counts[obj.classname] += 1
            if obj.classname not in cate_list:
                cate_list.append(obj.classname)

    cate_list = np.sort(cate_list)

    content = ''
    for x in cate_list:
        if counts[x] > 200:
            content += '%s %d'%(x, counts[x]) + "\n"
    # content = ''.join(['%s %d'%(x, counts[x]) + "\n" for x in cate_list])
    filename = os.path.join(SUNRGBD_DIR, "categories.txt")
    save(filename, content)


if __name__=='__main__':
    list_all()
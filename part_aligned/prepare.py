import os
import shutil

from cfg import settings
from os.path import join as pjoin


def prepare():
    prepare_train_val()


def prepare_train_val():
    data_dir = pjoin(settings.data_root, settings.prepared_market1501_name, 'train_val')
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir)
    else:
        return

    train_dir = pjoin(settings.data_root, settings.prepared_market1501_name, 'train')
    val_dir = pjoin(settings.data_root, settings.prepared_market1501_name, 'val')
    classes = os.listdir(train_dir)

    for classname in classes:
        imgnames_train = os.listdir(pjoin(train_dir, classname))
        imgnames_val = os.listdir(pjoin(val_dir, classname))
        if not os.path.isdir(pjoin(data_dir, classname)):
            os.makedirs(pjoin(data_dir, classname))
        for name in imgnames_train:
            src = pjoin(train_dir, classname, name)
            dst = pjoin(data_dir, classname, name)
            shutil.copy(src, dst)

        for name in imgnames_val:
            src = pjoin(val_dir, classname, name)
            dst = pjoin(data_dir, classname, name)
            shutil.copy(src, dst)

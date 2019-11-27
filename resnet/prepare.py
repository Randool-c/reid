import os
import shutil

from os.path import join as pjoin

from cfg import settings


market_root = pjoin(settings.data_root, 'Market-1501-v15.09.15')
target_root = pjoin(settings.data_root, settings.prepared_market1501_name)


def prepare():
    prepare_train_val()
    prepare_gallery()
    prepare_multi_query()
    prepare_query()


def prepare_train_val():
    trainpath = pjoin(target_root, 'train')
    valpath = pjoin(target_root, 'val')
    if not os.path.isdir(trainpath):
        os.makedirs(trainpath)
        os.makedirs(valpath)
    else:
        return

    for root, dirs, files in os.walk(pjoin(market_root, 'bounding_box_train')):
        assert len(dirs) == 0
        for imgname in files:
            if not os.path.splitext(imgname)[1] == '.jpg':
                continue

            pid = imgname.split('_')
            srcpath = pjoin(root, imgname)
            if not os.path.isdir(pjoin(trainpath, pid[0])):
                os.makedirs(pjoin(trainpath, pid[0]))
                os.makedirs(pjoin(valpath, pid[0]))
                dstpath = pjoin(valpath, pid[0], imgname)
            else:
                dstpath = pjoin(trainpath, pid[0], imgname)
            shutil.copy(srcpath, dstpath)


def prepare_query():
    querypath = pjoin(target_root, 'query')
    if not os.path.isdir(querypath):
        os.makedirs(querypath)
    else:
        return

    for root, dirs, files in os.walk(pjoin(market_root, 'query')):
        assert len(dirs) == 0
        for imgname in files:
            if not os.path.splitext(imgname)[1] == '.jpg':
                continue

            pid = imgname.split('_')
            srcpath = pjoin(root, imgname)
            if not os.path.isdir(pjoin(querypath, pid[0])):
                os.makedirs(pjoin(querypath, pid[0]))
            dstpath = pjoin(querypath, pid[0], imgname)
            shutil.copy(srcpath, dstpath)


def prepare_multi_query():
    querypath = pjoin(target_root, 'multi-query')
    if not os.path.isdir(querypath):
        os.makedirs(querypath)
    else:
        return

    for root, dirs, files in os.walk(pjoin(market_root, 'gt_bbox')):
        assert len(dirs) == 0
        for imgname in files:
            if not os.path.splitext(imgname)[1] == '.jpg':
                continue

            pid = imgname.split('_')
            srcpath = pjoin(root, imgname)
            if not os.path.isdir(pjoin(querypath, pid[0])):
                os.makedirs(pjoin(querypath, pid[0]))
            dstpath = pjoin(querypath, pid[0], imgname)
            shutil.copy(srcpath, dstpath)


def prepare_gallery():
    gallerypath = pjoin(target_root, 'gallery')
    if not os.path.isdir(gallerypath):
        os.makedirs(gallerypath)
    else:
        return

    for root, dirs, files in os.walk(pjoin(market_root, 'bounding_box_test')):
        assert len(dirs) == 0
        for imgname in files:
            if not os.path.splitext(imgname)[1] == '.jpg':
                continue

            pid = imgname.split('_')
            srcpath = pjoin(root, imgname)
            if not os.path.isdir(pjoin(gallerypath, pid[0])):
                os.makedirs(pjoin(gallerypath, pid[0]))

            dstpath = pjoin(gallerypath, pid[0], imgname)
            shutil.copy(srcpath, dstpath)


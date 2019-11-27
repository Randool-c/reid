import torch
import torch.nn as nn
import os
import random

from torch.utils.data import DataLoader, Dataset
from os.path import join as pjoin
from PIL import Image


class MyDataset(Dataset):
    def __init__(self, root, img_transforms):
        self.root = root
        self.classes = os.listdir(root)
        random.shuffle(self.classes)
        self.source = self.get_img_list()
        self.transforms = img_transforms

    def get_img_list(self):
        source = []
        for idx, classname in enumerate(self.classes):
            classsource = [(pjoin(self.root, classname, imgname), idx)
                           for imgname in os.listdir(pjoin(self.root, classname))]
            source += classsource
        return source

    def __getitem__(self, idx):
        path, label = self.source[idx]
        img = Image.open(path)

        img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.source)

    def shuffle(self):
        random.shuffle(self.classes)
        self.source = self.get_img_list()


class MyDataloader(DataLoader):
    def __init__(self, root, img_transforms, **kwargs):
        self.root = root
        self.dataset = MyDataset(root, img_transforms=img_transforms)
        super().__init__(self.dataset, **kwargs)

    def shuffle(self):
        self.dataset.shuffle()

    def get_img_paths(self):
        return [x[0] for x in self.dataset.source]

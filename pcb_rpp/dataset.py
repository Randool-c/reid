import os
import cv2
import json
import numpy as np

from PIL import Image
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader
from os.path import join as pjoin

from . import transforms


class MaskedDataset(Dataset):
    def __init__(self, root, matched_group_root, usage):
        assert usage in ['train', 'val', 'test']
        self.usage = usage
        self.root = root
        self.pic_to_box_dict = self.process(matched_group_root)
        if usage == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128)),
                transforms.RandomHorizontalFlip(),
                ToTensor(),
            ])
        elif usage == 'val':
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128)),
                ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128)),
                ToTensor()
            ])

        # self.dataset = datasets.ImageFolder(self.root, transform=self.transforms)
        self.group_to_box_dict = self.process(matched_group_root)
        self.groupnames = sorted(os.listdir(self.root))
        self.class_to_idx = {k: v for v, k in enumerate(self.groupnames)}
        self.source = []
        for idx, groupname in enumerate(self.groupnames):
            imgnames = os.listdir(pjoin(self.root, groupname))
            for imgname in imgnames:
                basename = os.path.splitext(imgname)[0]
                if groupname in self.group_to_box_dict and basename in self.group_to_box_dict[groupname]:
                    self.source.append((idx, basename))
        print(len(self))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        gidx, imgname = self.source[idx]
        gid = self.groupnames[gidx]
        img = Image.open(pjoin(self.root, gid, imgname + '.jpeg'))
        w, h = img.size
        facebox = self.group_to_box_dict[gid][imgname]['faceBox']
        bodybox = self.group_to_box_dict[gid][imgname]['bodyBox']
        cropy = max(facebox[3] - bodybox[1], 0)
        if facebox[3] - bodybox[1] < 0:
            print(facebox[3], bodybox[1])
        elif cropy >= h:
            cropy = 0
        img = img.crop([0, cropy, w, h])
        img = np.array(img)

        # if self.usage == 'train':
        #     if not os.path.isdir(pjoin('cropimgs', gid)):
        #         os.makedirs(pjoin('cropimgs', gid))
        #     if not os.path.isfile(pjoin('cropimgs', gid, imgname + '.jpg')):
        #         img.save(pjoin('cropimgs', gid, imgname + '.jpg'))
        img = self.transforms(img)
        return img, gidx

    def get_class_names(self):
        return self.groupnames

    def get_class_to_idx(self):
        return self.class_to_idx

    def get_img_paths(self):
        return [(pjoin(self.root, self.groupnames[x[0]], x[1]), self.groupnames[x[0]]) for x in self.source]

    def process(self, group_root):
        groupnames = os.listdir(group_root)
        d = {}
        for name in groupnames:
            with open(pjoin(group_root, name), 'r') as f:
                data = json.load(f)['data']
                f.close()

            for gid, items in data.items():
                for item in items:
                    pic_id = item['picId']
                    if 'faceBox' in item and 'bodyBox' in item:
                        if gid not in d:
                            d[gid] = {}
                        d[gid][pic_id] = item
        return d


class MaskedDataloader(DataLoader):
    def __init__(self, root, matched_group_root, usage, dataloader_args):
        assert usage in ['train', 'val', 'test']
        self.dataset = MaskedDataset(root, matched_group_root, usage)
        super().__init__(self.dataset, **dataloader_args)

    def get_class_names(self):
        return self.dataset.get_class_names()

    def get_class_to_idx(self):
        return self.dataset.get_class_to_idx()

    def get_img_paths(self):
        return self.dataset.get_img_paths()


class MarketDataloader(DataLoader):
    def __init__(self, root, usage, dataloader_args):
        assert usage in ['train', 'val', 'test']
        self.root = root
        if usage == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128)),
                transforms.RandomHorizontalFlip(),
                ToTensor(),
            ])
        elif usage == 'val':
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128)),
                ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128)),
                ToTensor()
            ])
        self.dataset = datasets.ImageFolder(self.root, transform=self.transforms)
        super().__init__(self.dataset, **dataloader_args)

    def get_class_names(self):
        return self.dataset.classes

    def get_class_to_idx(self):
        return self.dataset.class_to_idx

    def get_img_paths(self):
        return self.dataset.imgs

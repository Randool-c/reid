import torch
import torch.nn as nn
import os
import pickle
import numpy as np
import json

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from os.path import join as pjoin

from . import model
from . import prepare
from cfg import settings
from utils import flip


class ResnetSolver:
    def __init__(self, class_num):
        # self.model = model.PCBNet(class_num, 6)
        self.model = model.Resnet101(class_num)
        self.device = 'cuda:2' if torch.cuda.is_available() else 'cpu'
        # self.device = 'cpu'
        self.PCBepoch = 0

    def convert_to_rpp(self):
        self.model.convert_to_rpp()

    def load(self, state_dict):
        self.model.load_state_dict(state_dict)

    def save(self, path, info=None):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            **info
        }, path)

    def reset(self):
        self.PCBepoch = 0

    def train_resnet(self, dataloader_train, dataloader_val, opt, lr_scheduler, max_epochs=60,
                     saved_path='./resnet.model', output_path='./print.out'):
        criterion = nn.CrossEntropyLoss(reduction='mean')
        sm = nn.Softmax(dim=1)

        self.model.to(self.device)

        num_train_batches = len(dataloader_train)
        num_val_batches = len(dataloader_val)
        classes = dataloader_train.get_class_names()

        min_loss = float('inf')

        f = open(output_path, 'a')

        while self.PCBepoch < max_epochs:
            self.model.train()
            train_loss = 0
            train_acc = 0
            for (x, y) in dataloader_train:
                x = x.to(self.device)
                y = y.to(self.device)
                opt.zero_grad()
                outputs = self.model(x)
                loss = criterion(outputs, y)
                loss.backward()
                opt.step()

                train_loss += loss.item()
                pred = outputs.argmax(dim=1)
                train_acc += (pred == y).float().mean().item()
            train_acc /= num_train_batches
            train_loss /= num_train_batches

            val_loss = 0
            val_acc = 0
            self.model.eval()
            with torch.no_grad():
                for (x, y) in dataloader_val:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    opt.zero_grad()
                    outputs = self.model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    pred = outputs.argmax(dim=1)
                    val_acc += (pred == y).float().mean().item()
            val_acc /= num_val_batches
            val_loss /= num_val_batches

            if val_loss <= min_loss:
                self.save(saved_path, info={'num_classes': len(classes)})
                min_loss = val_loss

            info = 'epoch {}; train loss {:.4f}; train acc {:.4f}; val loss {:4f}; val acc {:.4f}'.format(
                self.PCBepoch, train_loss, train_acc, val_loss, val_acc
            )
            print(info)
            f.write(info + '\n')

            lr_scheduler.step()
            self.PCBepoch += 1
        print('complete training')
        f.close()
        self.reset()

    def extract(self, dataloader):
        self.model.eval()
        self.model.to(self.device)
        features = []
        with torch.no_grad():
            for (x, y) in dataloader:
                x = x.to(self.device)
                feature = self.model.output_feature(x)

                fnorm = torch.norm(feature, p=2, dim=1, keepdim=True)
                feature = feature.div(fnorm.expand_as(feature))
                print(feature.ndim)
                assert feature.ndim == 2
                features.append(feature)
        features = torch.cat(features, dim=0)
        return features


class MarketDataloader(DataLoader):
    def __init__(self, root, usage, dataloader_args):
        assert usage in ['train', 'val', 'test']
        self.root = root
        if usage == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif usage == 'val':
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128), interpolation=3),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128), interpolation=3),
                transforms.ToTensor()
            ])
        self.dataset = datasets.ImageFolder(self.root, transform=self.transforms)
        super().__init__(self.dataset, **dataloader_args)

    def get_class_names(self):
        return self.dataset.classes

    def get_class_to_idx(self):
        return self.dataset.class_to_idx

    def get_img_paths(self):
        return self.dataset.imgs


def get_ids(img_paths):
    camera_id = []
    labels = []
    for path, _ in img_paths:
        filename = os.path.basename(path)
        label = filename[:4]
        camera = filename.split('c')[1]
        if label[:2] == '-1':
            labels.append(-1)
        else:
            labels.append(int(label))
        camera_id.append(int(camera[0]))
    return camera_id, labels


class MaskedDataset(Dataset):
    def __init__(self, root, matched_group_root, usage):
        assert usage in ['train', 'val', 'test']
        self.root = root
        self.pic_to_box_dict = self.process(matched_group_root)
        if usage == 'train':
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128), interpolation=3),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        elif usage == 'val':
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128), interpolation=3),
                transforms.ToTensor()
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((384, 128), interpolation=3),
                transforms.ToTensor()
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

        if not os.path.isdir(pjoin('cropimgs', gid)):
            os.makedirs(pjoin('cropimgs', gid))
        if not os.path.isfile(pjoin('cropimgs', gid, imgname + '.jpg')):
            img.save(pjoin('cropimgs', gid, imgname + '.jpg'))
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


def train_resnet():
    prepare.prepare()

    batch_size = 64

    dataloader_train = MaskedDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'train'),
                                        pjoin(settings.data_root, 'matched_group'), 'train',
                                        {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16})
    # dataloader_train = MarketDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'train'), 'train',
    #                                     {'batch_size': batch_size, 'shuffle': True, 'num_workers': 16})
    dataloader_val = MaskedDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'val'),
                                      pjoin(settings.data_root, 'matched_group'), 'val',
                                      {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16})
    classes = dataloader_train.get_class_names()
    solver = ResnetSolver(len(classes))

    opt = torch.optim.SGD(solver.model.parameters(), lr=1e-2, weight_decay=5e-4, momentum=0.9, nesterov=True)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=40, gamma=0.1)
    solver.train_resnet(dataloader_train, dataloader_val, opt=opt, lr_scheduler=lr_scheduler, max_epochs=100,
                        saved_path=pjoin(settings.root, 'resnet', 'trained', 'resnet.model'))


def extract_features():
    feature_saved_path = pjoin(settings.root, 'resnet', 'extracted_features', 'features.pkl')

    batch_size = 64
    dataloader_gallery = MarketDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'gallery'),
                                          'test', {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16})
    dataloader_query = MarketDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'query'), 'test',
                                        {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16})

    saved_model = torch.load(pjoin(settings.root, 'resnet', 'trained', 'resnet.model'))
    solver = PCBRPPSolver(class_num=saved_model['num_classes'])
    solver.load(saved_model['model_state_dict'])

    feature_gallery = solver.extract(dataloader_gallery).cpu().numpy()
    feature_query = solver.extract(dataloader_query).cpu().numpy()
    # gallery_cameras, gallery_labels = get_ids(dataloader_gallery.get_img_paths())
    # query_cameras, query_labels = get_ids(dataloader_query.get_img_paths())
    gallery_labels = [item[1] for item in dataloader_gallery.get_img_paths()]
    query_labels = [item[1] for item in dataloader_query.get_img_paths()]
    gallery_cameras = np.zeros_like(gallery_labels)
    query_cameras = np.zeros_like(query_labels)

    pickle.dump({
        'gallery': {
            'features': feature_gallery,
            'cameras': gallery_cameras,
            'labels': gallery_labels
        },
        'query': {
            'features': feature_query,
            'cameras': query_cameras,
            'labels': query_labels
        }
    }, open(feature_saved_path, 'wb'))
    print('Finish! features saved in {}'.format(feature_saved_path))


def extract_features_masked():
    feature_saved_path = pjoin(settings.root, 'resnet', 'extracted_features', 'features.pkl')

    batch_size = 64
    # dataloader_gallery = MarketDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'gallery'),
    #                                       'test', {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16})
    dataloader_gallery = MaskedDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'gallery'),
                                          pjoin(settings.data_root, 'matched_group'), 'test',
                                          {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16})
    dataloader_query = MaskedDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'query'),
                                        pjoin(settings.data_root, 'matched_group'), 'test',
                                        {'batch_size': batch_size, 'shuffle': False, 'num_workers': 16})

    saved_model = torch.load(pjoin(settings.root, 'resnet', 'trained', 'resnet.model'))
    solver = ResnetSolver(class_num=saved_model['num_classes'])
    solver.load(saved_model['model_state_dict'])

    feature_gallery = solver.extract(dataloader_gallery).cpu().numpy()
    feature_query = solver.extract(dataloader_query).cpu().numpy()
    # gallery_cameras, gallery_labels = get_ids(dataloader_gallery.get_img_paths())
    # query_cameras, query_labels = get_ids(dataloader_query.get_img_paths())
    # groupnames = dataloader_gallery.get_class_names()
    classtoidx = dataloader_gallery.get_class_to_idx()
    gallery_labels = [item[1] for item in dataloader_gallery.get_img_paths()]
    gallery_labels = [classtoidx[x] for x in gallery_labels]
    query_labels = [item[1] for item in dataloader_query.get_img_paths()]
    query_labels = [classtoidx[x] for x in query_labels]
    gallery_cameras = np.zeros_like(gallery_labels)
    query_cameras = np.zeros_like(query_labels)

    pickle.dump({
        'gallery': {
            'features': feature_gallery,
            'cameras': gallery_cameras,
            'labels': gallery_labels
        },
        'query': {
            'features': feature_query,
            'cameras': query_cameras,
            'labels': query_labels
        }
    }, open(feature_saved_path, 'wb'))
    print('Finish! features saved in {}'.format(feature_saved_path))


def evaluate_single_orig(query_feature, query_label, query_camera, gallery_features, gallery_labels, gallery_cameras):
    assert query_feature.ndim == 1 and gallery_features.ndim == 2
    scores = gallery_features @ query_feature
    index = np.argsort(-scores)  # from large to small

    # good index
    query_index = np.argwhere(gallery_labels == query_label)
    camera_index = np.argwhere(gallery_cameras == query_camera)

    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gallery_labels == -1)
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    result = compute_mAP(index, good_index, junk_index)
    return result


def evaluate_single(query_feature, query_label, query_camera, gallery_features, gallery_labels, gallery_cameras):
    assert query_feature.ndim == 1 and gallery_features.ndim == 2
    scores = gallery_features @ query_feature
    index = np.argsort(-scores)  # from large to small

    # good index
    query_index = np.argwhere(gallery_labels == query_label)
    # camera_index = np.argwhere(gallery_cameras == query_camera)

    # good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    good_index = query_index
    # junk_index1 = np.argwhere(gallery_labels == -1)
    # junk_index2 = np.intersect1d(query_index, camera_index)
    # junk_index = np.append(junk_index2, junk_index1)
    junk_index = np.array([])

    result = compute_mAP(index, good_index, junk_index)
    return result


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = np.zeros_like(index)

    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc

    # remove junk indices
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find the indices in "index" of good indices
    n = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask).flatten()

    cmc[rows_good[0]:] = 1
    # compute average precision -- area under the PR curve
    d_recall = 1 / n
    for i in range(n):
        precision = (i + 1) / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i / rows_good[i]
        else:
            old_precision = 1
        ap += d_recall * (old_precision + precision) / 2
    return ap, cmc


def evaluate_features():
    feature_saved_path = pjoin(settings.root, 'resnet', 'extracted_features', 'features.pkl')
    features = pickle.load(open(feature_saved_path, 'rb'))

    gallery_features = features['gallery']['features']
    gallery_cameras = np.array(features['gallery']['cameras'])
    gallery_labels = np.array(features['gallery']['labels'])
    query_features = features['query']['features']
    query_cameras = np.array(features['query']['cameras'])
    query_labels = np.array(features['query']['labels'])

    ap = .0
    cmc = np.zeros_like(gallery_labels)
    total_num = len(query_labels)
    for i in range(len(query_labels)):
        print('processing {}; {} left'.format(i, total_num - i))
        ap_tmp, cmc_tmp = evaluate_single(query_features[i], query_labels[i], query_cameras[i],
                                          gallery_features, gallery_labels, gallery_cameras)
        if cmc_tmp[0] == -1:
            continue

        cmc = cmc + cmc_tmp
        ap += ap_tmp

    cmc = cmc.astype(float)
    cmc = cmc / len(query_labels)
    print('Rank 1: {}; Rank 5: {}; Rank 10: {}; mAP: {}'.format(cmc[0], cmc[4], cmc[9], ap / len(query_labels)))


def evaluate_resnet():
    prepare.prepare()

    # extract_features()
    extract_features_masked()
    evaluate_features()

import torch
import torch.nn as nn
import os
import pickle
import numpy as np

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from os.path import join as pjoin

from . import model
from . import prepare
from cfg import settings
from utils import flip


class PCBSolver:
    def __init__(self, class_num):
        self.model = model.PCBArcNet(class_num, 6)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    def make_opt(self, main_lr=None, classifier_lr=None, rpp_lr=None, lr_decay=0.1, lr_decay_step=40, momentum=0.9,
                 weight_decay=5e-4, use_nesterov_momentum=True):
        classifiers_params = list(self.model.classifiers.parameters())
        classifiers_params_ids = [id(x) for x in classifiers_params]
        pool_params = list(self.model.avgpool.parameters())
        pool_params_ids = [id(x) for x in pool_params]

        other_params = list(param for param in self.model.parameters()
                            if id(param) not in classifiers_params_ids and id(param) not in pool_params_ids)
        print(len(classifiers_params), len(pool_params), len(other_params), len(list(self.model.parameters())))

        params = []
        if main_lr is not None:
            params.append({'params': other_params, 'lr': main_lr})
        if classifier_lr is not None:
            params.append({'params': classifiers_params, 'lr': classifier_lr})
        if rpp_lr is not None:
            params.append({'params': pool_params, 'lr': rpp_lr})
        print('num params group ', len(params))
        opt = torch.optim.SGD(params, weight_decay=weight_decay, momentum=momentum, nesterov=use_nesterov_momentum)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=lr_decay_step, gamma=lr_decay)

        return opt, lr_scheduler

    def train(self, dataloader_train, dataloader_val, opt, lr_scheduler, max_epochs=60,
              saved_path='./pcb.model', output_path='./print.out'):
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
                outputs = self.model(x, y)
#                 print(outputs)
                loss = sum(criterion(part, y) for part in outputs)
                loss.backward()
                opt.step()

                train_loss += loss.item()
                pred = sum([sm(part) for part in outputs]).argmax(dim=1)
#                 print(pred)
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
                    outputs = self.model(x, y)
                    loss = sum(criterion(part, y) for part in outputs)
                    val_loss += loss.item()
                    pred = sum([sm(part) for part in outputs]).argmax(dim=1)
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
                xl = x.to(self.device)
                fl = self.model.output_feature(xl)
                xr = flip.flip_lr(xl)
                fr = self.model.output_feature(xr)
                feature = fl + fr
                assert feature.shape[1:] == (2048, 6)

                # normalize features on each part
                fnorm = torch.norm(feature, p=2, dim=1, keepdim=True) * np.sqrt(6)
                feature = feature.div(fnorm.expand_as(feature))
                feature = feature.view(-1, 2048 * 6)
                features.append(feature)
        features = torch.cat(features, dim=0)
        return features


class PCBRPPSolver(PCBSolver):
    def __init__(self, class_num):
        super().__init__(class_num)
        self.model.convert_to_rpp()


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


def train_pcb_rpp():
    prepare.prepare()

    batch_size = 64

    dataloader_train = MarketDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'train'), 'train',
                                        {'batch_size': batch_size, 'shuffle': True, 'num_workers': 16})
    dataloader_val = MarketDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'val'), 'val',
                                      {'batch_size': batch_size, 'shuffle': True, 'num_workers': 16})
    classes = dataloader_train.get_class_names()
    solver = PCBSolver(len(classes))

    opt, scheduler = solver.make_opt(
        main_lr=0.01, classifier_lr=0.1, rpp_lr=None, lr_decay=0.1, lr_decay_step=40, momentum=0.9, weight_decay=5e-4,
        use_nesterov_momentum=True
    )
    solver.train(dataloader_train, dataloader_val, opt=opt, lr_scheduler=scheduler, max_epochs=60,
                 saved_path=pjoin(settings.root, 'arc_pcb_rpp', 'trained', 'pcb.model'))

    solver.convert_to_rpp()
    opt, scheduler = solver.make_opt(
        main_lr=None, classifier_lr=None, rpp_lr=0.01, lr_decay=0.1, lr_decay_step=100, momentum=0.9, weight_decay=5e-4,
        use_nesterov_momentum=True
    )
    solver.train(dataloader_train, dataloader_val, opt=opt, lr_scheduler=scheduler, max_epochs=5,
                 saved_path=pjoin(settings.root, 'arc_pcb_rpp', 'trained', 'rpp.model'))

    opt, scheduler = solver.make_opt(
        main_lr=0.001, classifier_lr=0.01, rpp_lr=0.01, lr_decay=0.1, lr_decay_step=100, momentum=0.9,
        weight_decay=5e-4, use_nesterov_momentum=True)
    solver.train(dataloader_train, dataloader_val, opt=opt, lr_scheduler=scheduler, max_epochs=10,
                 saved_path=pjoin(settings.root, 'arc_pcb_rpp', 'trained', 'pcb_rpp.model'))


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


def extract_features():
    feature_saved_path = pjoin(settings.root, 'arc_pcb_rpp', 'extracted_features', 'features.pkl')

    batch_size = 64
    dataloader_gallery = MarketDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'gallery'),
                                          'test', {'batch_size': batch_size, 'shuffle': False, 'num_workers': 8})
    dataloader_query = MarketDataloader(pjoin(settings.data_root, settings.prepared_market1501_name, 'query'), 'test',
                                        {'batch_size': batch_size, 'shuffle': False, 'num_workers': 8})

    saved_model = torch.load(pjoin(settings.root, 'arc_pcb_rpp', 'trained', 'pcb_rpp.model'))
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
    feature_saved_path = pjoin(settings.root, 'arc_pcb_rpp', 'extracted_features', 'features.pkl')
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


def evaluate_pcb_rpp():
    prepare.prepare()

    # extract_features()
    evaluate_features()

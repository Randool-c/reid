import torch
import torch.nn as nn
import os
import pickle
import numpy as np

from torchvision import transforms

from . import model
from . import dataset
from . import prepare
from os.path import join as pjoin
from cfg import settings

pretrained_Inception_path = pjoin(settings.root, 'part_aligned', 'pretrained', 'bvlc_googlenet.caffemodel.pth')
pretrianed_CPM_path = pjoin(settings.root, 'part_aligned', 'pretrained', 'pose_iter_440000.caffemodel.pth')


class Solver:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.inception_v1_cpm(pretrained_Inception_path=pretrained_Inception_path,
                                            pretrained_CPM_path=pretrianed_CPM_path, initialize=True)
        self.cnt_iter = 0
        self.lr = None

    def load(self, path=None):
        if path is None:
            path = pjoin(pjoin(settings.root, 'part_aligned', 'trained',
                               sorted(os.listdir(pjoin(settings.root, 'part_aligned', 'trained')))[-1]))
        saved = torch.load(path)
        self.cnt_iter = saved['iters']
        self.model.load(saved['state_dict'])
        self.lr = saved['lr']

    def save(self, path):
        state_dict = self.model.save_dict()
        torch.save({
            'state_dict': state_dict,
            'iters': self.cnt_iter,
            'lr': self.lr
        }, path)

    def make_opt(self, init_lr, weight_decay, momentum, use_nesterov=True):
        params_bias = []
        params_weights = []
        for key, value in self.model.named_parameters():
            if value.requires_grad is False:
                continue
            if key[:-4] == 'bias':
                params_bias.append(value)
            else:
                params_weights.append(value)
        return torch.optim.SGD([
            {'params': params_bias, 'lr': init_lr * 2},
            {'params': params_weights, 'lr': init_lr}
        ], weight_decay=weight_decay, momentum=momentum, nesterov=use_nesterov)

    def train(self, dataloader_train, init_lr=0.01, lr_decay=0.2, weight_decay=2e-4, momentum=0.9, max_iters=75000,
              max_saved=10, trained_dir='trained', output_path='./print.out'):
        def update(optimizer, new_lr):
            for g in optimizer.param_groups:
                g['lr'] = new_lr

        criterion = TripletLoss(device=self.device)

        if self.lr is None:
            self.lr = init_lr
        opt = self.make_opt(self.lr, weight_decay, momentum, True)
        self.model.to(self.device)

        logfile = open(output_path, 'a')

        accumu_loss = 0
        while self.cnt_iter < max_iters:
            for x, y in dataloader_train:
                y = y.to(self.device)
                x = x.to(self.device)
                out = self.model(x)
                loss = criterion(out, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

                accumu_loss += loss.item()

                print('iter {}; loss {}'.format(self.cnt_iter, loss.item()))

                self.cnt_iter += 1

                # cnt_iter % 20000 == 0: update opt
                if self.cnt_iter % 20000 == 0:
                    self.lr *= lr_decay
                    update(opt, self.lr)

                if self.cnt_iter % 200 == 0:
                    accumu_loss /= 200
                    loginfo = 'iter {}; loss {:.6f}'.format(self.cnt_iter, accumu_loss)
                    logfile.write(loginfo + '\n')
                    logfile.flush()
                    print(loginfo)
                    accumu_loss = 0

                    self.save(pjoin(trained_dir, 'checkpoint_iter_' + str(self.cnt_iter) + '.pth'))
                    if os.path.isfile(pjoin(
                            trained_dir, 'checkpoint_iter_' + str(self.cnt_iter - 200 * max_saved) + '.pth')):
                        os.remove(pjoin(
                            trained_dir, 'checkpoint_iter_' + str(self.cnt_iter - 200 * max_saved) + '.pth'))

            dataloader_train.shuffle()

        self.save(pjoin(trained_dir, 'checkpoint_iter_final.pth'))
        logfile.close()
        print('complete training')

    def extract(self, dataloader):
        features = []
        self.model.eval()
        self.model.to(self.device)
        num_batches = len(dataloader)
        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                print('processing {} batch; {} batches left'.format(i, num_batches - i))
                x = x.to(self.device)
                out = self.model(x)
                features.append(out)
        features = torch.cat(features, dim=0)
        return features


class TripletLoss(nn.Module):
    def __init__(self, margin=0.2, device='cuda'):
        super().__init__()
        self.margin = margin
        self.device = device

    def get_pdist(self, inputs):
        inner_product = inputs @ inputs.t()  # compute distances of pairs
        square_norm = inner_product.diag()
        left_item = square_norm.unsqueeze(dim=1)
        right_item = square_norm.unsqueeze(dim=0)
        pdist = left_item + right_item - 2 * inner_product
        return pdist

    def get_mask(self, y):
        """return a mask to filter elements with i = j or i = k or j = k or not (i == j and i != k)"""

        batch_size = len(y)
        cube_shape = (batch_size, batch_size, batch_size)

        # i != j and j != k and i != k
        not_equal_square = 1 - torch.eye(batch_size).to(self.device)
        not_equal_ij = not_equal_square.unsqueeze(dim=2).expand(*cube_shape)
        not_equal_ik = not_equal_square.unsqueeze(dim=1).expand(*cube_shape)
        not_equal_jk = not_equal_square.unsqueeze(dim=0).expand(*cube_shape)
        not_equal = not_equal_ij * not_equal_jk * not_equal_ik

        # label[i] == label[j] and label[i] != label[k]
        equal_label = (y.unsqueeze(dim=0) == y.unsqueeze(dim=1)).float()
        equal_ij = equal_label.unsqueeze(dim=2).expand(*cube_shape)
        equal_ik = equal_label.unsqueeze(dim=1).expand(*cube_shape)
        valid_labels = equal_ij * (1 - equal_ik)
        mask = not_equal * valid_labels
        return mask

    def forward(self, inputs, y):
        """compute the triplet loss
            @:param inputs: a batch of the net's output features. [batch_size, feature_dim]
            @:param y: labels
        """

        assert inputs.ndim == 2
        batch_size = len(y)
        cube_shape = (batch_size, batch_size, batch_size)
        pdist = self.get_pdist(inputs)
        mask = self.get_mask(y)
        num_valid = mask.sum()

        dist_ij = pdist.unsqueeze(dim=2).expand(*cube_shape)
        dist_ik = pdist.unsqueeze(dim=1).expand(*cube_shape)
        loss_matrix = torch.max((dist_ij - dist_ik + self.margin) * mask, torch.zeros(*cube_shape, device=self.device))
        loss = loss_matrix.sum() / num_valid
        return loss


def train_partalign():
    prepare.prepare()

    batch_size = 180
    train_root = pjoin(settings.data_root, settings.prepared_market1501_name, 'train_val')

    dataloader_train = dataset.MyDataloader(train_root, img_transforms=transforms.Compose([
        transforms.Resize((160, 80), interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]), batch_size=batch_size, num_workers=8, shuffle=False)

    solver = Solver()
    solver.train(dataloader_train, init_lr=0.01, weight_decay=2e-4, momentum=0.9, max_iters=75000, max_saved=20,
                 lr_decay=0.2, trained_dir=pjoin(settings.root, 'part_aligned', 'trained'),
                 output_path=pjoin(settings.root, 'part_aligned', 'print.out'))


def get_ids(img_paths):
    camera_id = []
    labels = []
    for path in img_paths:
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
    feature_saved_path = pjoin(settings.root, 'part_aligned', 'extracted_features', 'features.pkl')
    gallery_root = pjoin(settings.data_root, settings.prepared_market1501_name, 'gallery')
    query_root = pjoin(settings.data_root, settings.prepared_market1501_name, 'query')

    batch_size = 256
    dataloader_gallery = dataset.MyDataloader(gallery_root, img_transforms=transforms.Compose([
        transforms.Resize((160, 80), interpolation=2),
        transforms.ToTensor()
    ]), batch_size=batch_size, num_workers=8, shuffle=False)
    dataloader_query = dataset.MyDataloader(query_root, img_transforms=transforms.Compose([
        transforms.Resize((160, 80), interpolation=2),
        transforms.ToTensor()
    ]), batch_size=batch_size, num_workers=8, shuffle=False)

    solver = Solver()
    solver.load()

    feature_gallery = solver.extract(dataloader_gallery).cpu().numpy()
    feature_query = solver.extract(dataloader_query).cpu().numpy()
    gallery_cameras, gallery_labels = get_ids(dataloader_gallery.get_img_paths())
    query_cameras, query_labels = get_ids(dataloader_query.get_img_paths())

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


def evaluate_single(query_feature, query_label, query_camera, gallery_features, gallery_labels, gallery_cameras):
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
    feature_saved_path = pjoin(settings.root, 'part_aligned', 'extracted_features', 'features.pkl')
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


def evaluate_part_aligned():
    extract_features()
    evaluate_features()

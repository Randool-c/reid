import torch
import torch.nn as nn

from torchvision import models

import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class Inception_base(nn.Module):
    def __init__(self, depth_dim, input_size, config, dilation=1):
        super(Inception_base, self).__init__()

        self.depth_dim = depth_dim

        self._1x1 = nn.Conv2d(input_size, out_channels=config[0][0], kernel_size=1, stride=1, padding=0)
        self._3x3_reduce = nn.Conv2d(input_size, out_channels=config[1][0], kernel_size=1, stride=1, padding=0)
        self._3x3 = nn.Conv2d(config[1][0], config[1][1], kernel_size=3, stride=1, padding=1 * dilation,
                              dilation=dilation)
        self._5x5_reduce = nn.Conv2d(input_size, out_channels=config[2][0], kernel_size=1, stride=1, padding=0)
        self._5x5 = nn.Conv2d(config[2][0], config[2][1], kernel_size=5, stride=1, padding=2 * dilation,
                              dilation=dilation)

        self.max_pool_1 = nn.MaxPool2d(kernel_size=config[3][0], stride=1, padding=1 * dilation)
        self.pool_proj = nn.Conv2d(input_size, out_channels=config[3][1], kernel_size=1, stride=1, padding=0)

    def forward(self, input):
        output1 = F.relu(self._1x1(input))

        output2 = F.relu(self._3x3_reduce(input))
        output2 = F.relu(self._3x3(output2))

        output3 = F.relu(self._5x5_reduce(input))
        output3 = F.relu(self._5x5(output3))

        output4 = F.relu(self.pool_proj(self.max_pool_1(input)))

        return torch.cat([output1, output2, output3, output4], dim=self.depth_dim)


class Inception_v1(nn.Module):
    def __init__(self, pretrained_path, num_features=512, dilation=1, initialize=True, feat_type='4e'):
        super(Inception_v1, self).__init__()
        self.dilation = dilation
        self.pretrained = pretrained_path
        self.feat_type = feat_type

        # conv2d0
        self.conv1__7x7_s2 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)
        self.lrn1 = nn.CrossMapLRN2d(5, 0.0001, 0.75, 1)

        # conv2d1
        self.conv2__3x3_reduce = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)

        # conv2d2
        self.conv2__3x3 = nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.lrn3 = nn.CrossMapLRN2d(5, 0.0001, 0.75, 1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        self.inception_3a = Inception_base(1, 192, [[64], [96, 128], [16, 32], [3, 32]])  # 3a
        self.inception_3b = Inception_base(1, 256, [[128], [128, 192], [32, 96], [3, 64]])  # 3b
        if self.dilation == 1:
            self.max_pool_inc3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)

        pool_kernel = self.dilation * 2 + 1
        self.inception_4a = Inception_base(1, 480, [[192], [96, 208], [16, 48], [pool_kernel, 64]],
                                           dilation=self.dilation)  # 4a
        self.inception_4b = Inception_base(1, 512, [[160], [112, 224], [24, 64], [pool_kernel, 64]],
                                           dilation=self.dilation)  # 4b
        self.inception_4c = Inception_base(1, 512, [[128], [128, 256], [24, 64], [pool_kernel, 64]],
                                           dilation=self.dilation)  # 4c
        self.inception_4d = Inception_base(1, 512, [[112], [144, 288], [32, 64], [pool_kernel, 64]],
                                           dilation=self.dilation)  # 4d
        self.inception_4e = Inception_base(1, 528, [[256], [160, 320], [32, 128], [pool_kernel, 128]],
                                           dilation=self.dilation)  # 4e

        if self.feat_type == '5b':
            assert self.dilation == 1
            pool_kernel = self.dilation * 2 + 1

            self.inception_5a = Inception_base(1, 832, [[256], [160, 320], [32, 128], [pool_kernel, 128]],
                                               dilation=self.dilation)  # 5a
            self.inception_5b = Inception_base(1, 832, [[384], [192, 384], [48, 128], [pool_kernel, 128]],
                                               dilation=self.dilation)  # 5b
            self.input_feat = nn.Conv2d(1024, num_features, (1, 1), (1, 1), (0, 0))
        else:
            self.input_feat = nn.Conv2d(832, num_features, (1, 1), (1, 1), (0, 0))
            self.bn = nn.BatchNorm2d(num_features, affine=False)

        if initialize:
            self.init_pretrained()

    def forward(self, input):

        output = self.max_pool1(F.relu(self.conv1__7x7_s2(input)))
        output = self.lrn1(output)

        output = F.relu(self.conv2__3x3_reduce(output))
        output = F.relu(self.conv2__3x3(output))
        output = self.max_pool3(self.lrn3(output))

        output = self.inception_3a(output)
        output = self.inception_3b(output)
        if self.dilation == 1:
            output = self.max_pool_inc3(output)

        output = self.inception_4a(output)
        output = self.inception_4b(output)
        output = self.inception_4c(output)
        output = self.inception_4d(output)
        output = self.inception_4e(output)

        if self.feat_type == '5b':
            output = self.inception_5a(output)
            output = self.inception_5b(output)
            output = nn.AdaptiveAvgPool2d(1)(output)
            output = self.input_feat(output)
            output = F.normalize(output, dim=1)
            output = output.squeeze()
        else:
            output = self.input_feat(output)
            output = self.bn(output)

        return output

    def init_pretrained(self):
        state_dict = torch.load(self.pretrained)
        state_dict['input_feat.weight'] = nn.init.xavier_uniform_(self.input_feat.weight).detach()
        state_dict['input_feat.bias'] = torch.zeros(self.input_feat.bias.size())
        if not self.feat_type == '5b':
            # TODO
            state_dict['bn.running_mean'] = self.bn.running_mean
            state_dict['bn.running_var'] = self.bn.running_var

        model_dict = {}
        for k, v in state_dict.items():
            for l, p in self.state_dict().items():
                if k.replace("/", ".") in l.replace("__", ".").replace("._", "."):
                    model_dict[l] = torch.from_numpy(np.array(v)).view_as(p)

                #        self.load_state_dict(model_dict, strict=False)
        self.load_state_dict(model_dict, strict=True)
        """
        print('inception v1 pretrained model loaded!')
        print('difference:--------------------------')
        print(set(self.state_dict().keys()).symmetric_difference(model_dict.keys()))
        print('inception v1 pretrained model loaded!')
        """

    def save_dict(self):
        return self.state_dict()

    def load(self, checkpoint):
        checkpoint.pop('epoch')
        if torch.__version__ == '0.4.0':
            checkpoint.pop('bn.num_batches_tracked', None)
        self.load_state_dict(checkpoint)


class CompactBilinearPooling(nn.Module):
    """
    Compute compact bilinear pooling over two bottom inputs.

    Args:

        output_dim: output dimension for compact bilinear pooling.

        sum_pool: (Optional) If True, sum the output along height and width
                  dimensions and return output shape [batch_size, output_dim].
                  Otherwise return [batch_size, height, width, output_dim].
                  Default: True.

        rand_h_1: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_1`
                  if is None.

        rand_s_1: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_1`. Automatically generated from `seed_s_1` if is
                  None.

        rand_h_2: (Optional) an 1D numpy array containing indices in interval
                  `[0, output_dim)`. Automatically generated from `seed_h_2`
                  if is None.

        rand_s_2: (Optional) an 1D numpy array of 1 and -1, having the same shape
                  as `rand_h_2`. Automatically generated from `seed_s_2` if is
                  None.
    """

    def __init__(self, input_dim1, input_dim2, output_dim,
                 sum_pool=True, cuda=True,
                 rand_h_1=None, rand_s_1=None, rand_h_2=None, rand_s_2=None):
        super(CompactBilinearPooling, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.output_dim = output_dim
        self.sum_pool = sum_pool

        if rand_h_1 is None:
            np.random.seed(1)
            rand_h_1 = np.random.randint(output_dim, size=self.input_dim1)
        if rand_s_1 is None:
            np.random.seed(3)
            rand_s_1 = 2 * np.random.randint(2, size=self.input_dim1) - 1

        self.sparse_sketch_matrix1 = nn.Parameter(self.generate_sketch_matrix(
            rand_h_1, rand_s_1, self.output_dim), requires_grad=False)

        if rand_h_2 is None:
            np.random.seed(5)
            rand_h_2 = np.random.randint(output_dim, size=self.input_dim2)
        if rand_s_2 is None:
            np.random.seed(7)
            rand_s_2 = 2 * np.random.randint(2, size=self.input_dim2) - 1

        self.sparse_sketch_matrix2 = nn.Parameter(self.generate_sketch_matrix(
            rand_h_2, rand_s_2, self.output_dim), requires_grad=False)

        if cuda:
            self.sparse_sketch_matrix1 = self.sparse_sketch_matrix1.cuda()
            self.sparse_sketch_matrix2 = self.sparse_sketch_matrix2.cuda()

    def forward(self, bottom1, bottom2):
        assert bottom1.size(1) == self.input_dim1 and \
            bottom2.size(1) == self.input_dim2

        batch_size, _, height, width = bottom1.size()

        bottom1_flat = bottom1.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim1)
        bottom2_flat = bottom2.permute(0, 2, 3, 1).contiguous().view(-1, self.input_dim2)

        sketch_1 = bottom1_flat.mm(self.sparse_sketch_matrix1)
        sketch_2 = bottom2_flat.mm(self.sparse_sketch_matrix2)

        fft1 = torch.fft(torch.cat((sketch_1.unsqueeze(-1), torch.zeros(sketch_1.size()).unsqueeze(-1).cuda()), -1), 1)
        fft2 = torch.fft(torch.cat((sketch_2.unsqueeze(-1), torch.zeros(sketch_2.size()).unsqueeze(-1).cuda()), -1), 1)

        fft1_real = fft1[..., 0]
        fft1_imag = fft1[..., 1]
        fft2_real = fft2[..., 0]
        fft2_imag = fft2[..., 1]

        temp_rr, temp_ii = fft1_real.mul(fft2_real), fft1_imag.mul(fft2_imag)
        temp_ri, temp_ir = fft1_real.mul(fft2_imag), fft1_imag.mul(fft2_real)
        fft_product_real = temp_rr - temp_ii
        fft_product_imag = temp_ri + temp_ir

        cbp_flat = torch.ifft(torch.cat((fft_product_real.unsqueeze(-1), fft_product_imag.unsqueeze(-1)), -1), 1)
        cbp_flat = cbp_flat[..., 0]

        cbp = cbp_flat.view(batch_size, height, width, self.output_dim)*self.output_dim

        if self.sum_pool:
            cbp = cbp.sum(dim=1).sum(dim=1)
        else:
            cbp = cbp.permute(0,3,1,2)

        return cbp

    @staticmethod
    def generate_sketch_matrix(rand_h, rand_s, output_dim):
        """
        Return a sparse matrix used for tensor sketch operation in compact bilinear
        pooling
        Args:
            rand_h: an 1D numpy array containing indices in interval `[0, output_dim)`.
            rand_s: an 1D numpy array of 1 and -1, having the same shape as `rand_h`.
            output_dim: the output dimensions of compact bilinear pooling.
        Returns:
            a sparse matrix of shape [input_dim, output_dim] for tensor sketch.
        """

        # Generate a sparse matrix for tensor count sketch
        rand_h = rand_h.astype(np.int64)
        rand_s = rand_s.astype(np.float32)
        assert(rand_h.ndim == 1 and rand_s.ndim ==
               1 and len(rand_h) == len(rand_s))
        assert(np.all(rand_h >= 0) and np.all(rand_h < output_dim))

        input_dim = len(rand_h)
        indices = np.concatenate((np.arange(input_dim)[..., np.newaxis],
                                  rand_h[..., np.newaxis]), axis=1)
        indices = torch.from_numpy(indices)
        rand_s = torch.from_numpy(rand_s)
        sparse_sketch_matrix = torch.sparse.FloatTensor(
            indices.t(), rand_s, torch.Size([input_dim, output_dim]))
        return sparse_sketch_matrix.to_dense().cuda()


class CPM(nn.Module):
    def __init__(self, depth_dim, pretrained_path, num_features_part=128, use_relu=False, dilation=1, initialize=True):
        super(CPM, self).__init__()
        self.pretrained = pretrained_path

        self.depth_dim = depth_dim
        self.use_relu = use_relu
        self.dilation = dilation

        self.conv1_1 = nn.Conv2d(3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1_stage1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv2_1 = nn.Conv2d(64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2_stage1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv3_1 = nn.Conv2d(128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.pool3_stage1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        self.conv4_1 = nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3_CPM = nn.Conv2d(512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_4_CPM = nn.Conv2d(256, out_channels=128, kernel_size=3, stride=1, padding=1)

        # Stage1
        # limbs
        self.conv5_1_CPM_L1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_2_CPM_L1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_3_CPM_L1 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_4_CPM_L1 = nn.Conv2d(128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_5_CPM_L1 = nn.Conv2d(512, out_channels=38, kernel_size=1, stride=1, padding=0)
        # joints
        self.conv5_1_CPM_L2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_2_CPM_L2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_3_CPM_L2 = nn.Conv2d(128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv5_4_CPM_L2 = nn.Conv2d(128, out_channels=512, kernel_size=1, stride=1, padding=0)
        self.conv5_5_CPM_L2 = nn.Conv2d(512, out_channels=19, kernel_size=1, stride=1, padding=0)

        if self.dilation == 1:
            self.concat_stage3_pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True)

        # Stage2
        # limbs
        self.Mconv1_stage2_L1 = nn.Conv2d(185, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage2_L1 = nn.Conv2d(128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage2_L1 = nn.Conv2d(128, out_channels=38, kernel_size=1, stride=1, padding=0)
        # joints
        self.Mconv1_stage2_L2 = nn.Conv2d(185, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv2_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv3_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv4_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv5_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=7, stride=1, padding=3)
        self.Mconv6_stage2_L2 = nn.Conv2d(128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.Mconv7_stage2_L2 = nn.Conv2d(128, out_channels=19, kernel_size=1, stride=1, padding=0)

        self.pose1 = nn.Conv2d(185, out_channels=num_features_part, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features_part, affine=False)

        if initialize:
            self.init_pretrained()

    def forward(self, inputs):
        # data rescale
        output1 = 0.0039 * inputs

        output1 = F.relu(self.conv1_1(output1))
        output1 = F.relu(self.conv1_2(output1))
        output1 = self.pool1_stage1(output1)

        output1 = F.relu(self.conv2_1(output1))
        output1 = F.relu(self.conv2_2(output1))
        output1 = self.pool2_stage1(output1)

        output1 = F.relu(self.conv3_1(output1))
        output1 = F.relu(self.conv3_2(output1))
        output1 = F.relu(self.conv3_3(output1))
        output1 = F.relu(self.conv3_4(output1))
        output1 = self.pool3_stage1(output1)

        output1 = F.relu(self.conv4_1(output1))
        output1 = F.relu(self.conv4_2(output1))
        output1 = F.relu(self.conv4_3_CPM(output1))
        output1 = F.relu(self.conv4_4_CPM(output1))

        output2_1 = F.relu(self.conv5_1_CPM_L1(output1))
        output2_1 = F.relu(self.conv5_2_CPM_L1(output2_1))
        output2_1 = F.relu(self.conv5_3_CPM_L1(output2_1))
        output2_1 = F.relu(self.conv5_4_CPM_L1(output2_1))
        output2_1 = self.conv5_5_CPM_L1(output2_1)

        output2_2 = F.relu(self.conv5_1_CPM_L2(output1))
        output2_2 = F.relu(self.conv5_2_CPM_L2(output2_2))
        output2_2 = F.relu(self.conv5_3_CPM_L2(output2_2))
        output2_2 = F.relu(self.conv5_4_CPM_L2(output2_2))
        output2_2 = self.conv5_5_CPM_L2(output2_2)

        output2 = torch.cat([output2_1, output2_2, output1], dim=self.depth_dim)

        output3_1 = F.relu(self.Mconv1_stage2_L1(output2))
        output3_1 = F.relu(self.Mconv2_stage2_L1(output3_1))
        output3_1 = F.relu(self.Mconv3_stage2_L1(output3_1))
        output3_1 = F.relu(self.Mconv4_stage2_L1(output3_1))
        output3_1 = F.relu(self.Mconv5_stage2_L1(output3_1))
        output3_1 = F.relu(self.Mconv6_stage2_L1(output3_1))
        output3_1 = self.Mconv7_stage2_L1(output3_1)

        output3_2 = F.relu(self.Mconv1_stage2_L2(output2))
        output3_2 = F.relu(self.Mconv2_stage2_L2(output3_2))
        output3_2 = F.relu(self.Mconv3_stage2_L2(output3_2))
        output3_2 = F.relu(self.Mconv4_stage2_L2(output3_2))
        output3_2 = F.relu(self.Mconv5_stage2_L2(output3_2))
        output3_2 = F.relu(self.Mconv6_stage2_L2(output3_2))
        output3_2 = self.Mconv7_stage2_L2(output3_2)

        output3 = torch.cat([output3_1, output3_2, output1], dim=self.depth_dim)
        if self.dilation == 1:
            output3 = self.concat_stage3_pool(output3)
        output3 = self.pose1(output3)
        output3 = self.bn(output3)
        if self.use_relu:
            output3 = F.relu(output3)

        return output3

    def init_pretrained(self):
        state_dict = torch.load(self.pretrained)
        state_dict['pose1.weight'] = nn.init.xavier_uniform_(self.pose1.weight).detach()
        state_dict['pose1.bias'] = torch.zeros(self.pose1.bias.size())
        # TODO
        state_dict['bn.running_mean'] = self.bn.running_mean
        state_dict['bn.running_var'] = self.bn.running_var

        model_dict = {}
        for k, v in state_dict.items():
            for l, p in self.state_dict().items():
                if k in l:
                    model_dict[l] = torch.from_numpy(np.array(v)).view_as(p)

        self.load_state_dict(model_dict, strict=True)
        print('cpm pretrained model loaded!')


class Bilinear_Pooling(nn.Module):

    def __init__(self, num_feat1=512, num_feat2=128, num_feat_out=512):
        super(Bilinear_Pooling, self).__init__()
        self.num_feat1 = num_feat1
        self.num_feat2 = num_feat2
        self.num_feat_out = num_feat_out

        self.cbp = CompactBilinearPooling(self.num_feat1, self.num_feat2, self.num_feat_out, sum_pool=True)

    def forward(self, input1, input2):
        assert (self.num_feat1 == input1.shape[1])
        assert (self.num_feat2 == input2.shape[1])
        output = self.cbp(input1, input2)
        output = F.normalize(output, dim=1)
        output = output.squeeze()

        return output


class Inception_v1_cpm(nn.Module):
    def __init__(self, pretrained_Inception_path, pretrained_CPM_path, num_features=512, use_bn=True, use_relu=False,
                 dilation=1, initialize=True):
        super(Inception_v1_cpm, self).__init__()
        num_features_app, num_features_part = (512, 128)
        self.app_feat_extractor = Inception_v1(
            pretrained_Inception_path, num_features=num_features_app, dilation=dilation, initialize=initialize)
        self.part_feat_extractor = CPM(
            1, pretrained_CPM_path, num_features_part=num_features_part, dilation=dilation, initialize=initialize)
        self.pooling = Bilinear_Pooling(num_feat1=num_features_app, num_feat2=num_features_part,
                                        num_feat_out=num_features)

    def forward(self, inputs):
        output_app = self.app_feat_extractor(inputs)
        output_part = self.part_feat_extractor(inputs)
        return self.pooling(output_app, output_part)

    def save_dict(self):
        state_dict = {'app_state_dict': self.app_feat_extractor.state_dict(),
                      'part_state_dict': self.part_feat_extractor.state_dict()}
        return state_dict

    def load(self, checkpoint):
        self.app_feat_extractor.load_state_dict(checkpoint['app_state_dict'])
        self.part_feat_extractor.load_state_dict(checkpoint['part_state_dict'])

    def init_pretrained(self):
        self.app_feat_extractor.init_pretrained()
        self.part_feat_extractor.init_pretrained()


def inception_v1_cpm(pretrained_Inception_path, pretrained_CPM_path, features=512, use_relu=False, dilation=1,
                     initialize=True):
    model = Inception_v1_cpm(
        pretrained_Inception_path=pretrained_Inception_path, pretrained_CPM_path=pretrained_CPM_path,
        num_features=features, use_relu=use_relu, dilation=dilation, initialize=initialize)

    return model

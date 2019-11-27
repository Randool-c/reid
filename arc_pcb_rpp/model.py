import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torchvision import models

# from cfg import settings


def weights_init_kaiming(module):
    """kaiming weights initialization"""

    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(module.bias.data, .0)
    elif isinstance(module, nn.BatchNorm1d):
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, .0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        nn.init.constant_(module.bias.data, 0)


def weights_init_classifier(module):
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight.data, std=0.001)
        nn.init.constant_(module.bias.data, 0.0)


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, relu=True, num_bottleneck=256, m=0.5, s=30.0):
        """constructor for ClassBlock.
            @:param num_bottleneck: number of dimensions of the output of the pooling layer.
        """
        super().__init__()

        self.m = m
        self.s = s
        self.class_num = class_num

        blocks = [
            nn.Conv2d(input_dim, num_bottleneck, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_bottleneck)
        ]
        if relu:
            blocks.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*blocks)
        self.conv.apply(weights_init_kaiming)

        self.classifier = nn.Parameter(torch.FloatTensor(class_num, num_bottleneck))
        nn.init.xavier_uniform_(self.classifier)

        self.cos_m = np.cos(self.m)
        self.sin_m = np.sin(self.m)
        # self.threshold = -np.cos(self.m)
        # self.mm = self.sin_m * m
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # self.classifier = nn.Linear(num_bottleneck, class_num)
        # self.classifier.apply(weights_init_classifier)

    def forward(self, x, y):
        x = self.conv(x)
        x = x.squeeze()
        # x = self.classifier(x)

        cosine = F.linear(F.normalize(x), F.normalize(self.classifier))
        sine = torch.sqrt((1.0 - cosine * cosine).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        # phi = torch.where(cosine > self.threshold, phi, cosine - self.mm)

        # onehot = torch.zeros_like(cosine).to(self.device)
        # onehot.scatter_(1, y.view(-1, 1).long(), 1)
        onehot = torch.eye(self.class_num)[y.long()].to(self.device)
        output = (onehot * phi) + ((1.0 - onehot) * cosine)
        output = output * self.s
        return output


class RPPLayer(nn.Module):
    def __init__(self, num_part=6, dim_backbone_out=2048):
        super().__init__()
        self.part = num_part

        self.conv_block = nn.Conv2d(dim_backbone_out, num_part, kernel_size=1, bias=False)
        self.conv_block.apply(weights_init_kaiming)

        self.norm_block = nn.Sequential(
            nn.BatchNorm2d(dim_backbone_out),
            nn.ReLU(inplace=True)
        )
        self.norm_block.apply(weights_init_kaiming)

        self.softmax = nn.Softmax(dim=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        w = self.conv_block(x)
        prob = self.softmax(w)  # chanel 1 indicates the probability of each f belonging to a part
        y = []
        for i in range(self.part):
            y_i = torch.mul(x, prob[:, [i], :, :])
            y_i = self.norm_block(y_i)
            y_i = self.avgpool(y_i)
            y.append(y_i)
        out = torch.cat(y, dim=2)
        return out


class PCBArcNet(nn.Module):
    def __init__(self, class_num, pool_part=6, m=0.5, s=30.0):
        """constructor for PCBArcNet
            @:param m: margin
            @:param s: scale of the features
        """
        super().__init__()

        self.part = pool_part

        resnet = models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)
        self.dim_backbone_out = 2048
        self.dim_bottleneck = 256

        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # remove the GAP layer and fc layer
        self.avgpool = nn.AdaptiveAvgPool2d((self.part, 1))
        self.dropout = nn.Dropout(p=0.5)

        self.classifiers = nn.ModuleList([ClassBlock(self.dim_backbone_out, class_num=class_num,
                                                     num_bottleneck=self.dim_bottleneck, relu=True)
                                          for _ in range(self.part)])
        for classifier in self.classifiers:
            classifier.apply(weights_init_classifier)

        print(self)

    def forward(self, x, y):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = self.dropout(x)

        predict = []
        for i in range(self.part):
            part = x[:, :, [i], :]
            predict.append(self.classifiers[i](part, y))
        return predict

    def output_feature(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        return x.reshape(-1, self.dim_backbone_out, self.part)

    def convert_to_rpp(self):
        self.avgpool = RPPLayer(self.part, self.dim_backbone_out)
        return self

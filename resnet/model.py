import torch
import torch.nn as nn
import numpy as np

from torchvision import models
from collections import OrderedDict

# from cfg import settings


class Resnet101(nn.Module):
    def __init__(self, class_num):
        super().__init__()
        model = models.resnet101(pretrained=True)

        extractor = OrderedDict()
        for k, v in model.named_children():
            if k == 'fc':
                break

            extractor[k] = v
        self.extractor = nn.Sequential(extractor)
        for k, v in model.named_parameters():
            if k in self.extractor:
                self.extractor[k].load_state_dict(v)
        # self.extractor = extractor

        self.fc = nn.Linear(2048, class_num, bias=True)

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def output_feature(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        return x

import torch
import torch.nn as nn

from torchvision import models


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


class Net(nn.Module):
    def __init__(self, class_num, part=8, bottleneck_channels=256):
        super().__init__()

        self.part = part

        resnet = models.resnet50(pretrained=True)
        resnet.layer4[0].downsample[0].stride = (1, 1)
        resnet.layer4[0].conv2.stride = (1, 1)

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.bottleneck_channels = bottleneck_channels

        self.backbone_out_channels = 2048
        self.part_channels = int(self.backbone_out_channels / part)

        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(self.part_channels, bottleneck_channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(self.part_channels),
            nn.ReLU(inplace=True)
        )
        self.bottleneck_conv.apply(weights_init_kaiming)

        # self.classifier = nn.ModuleList()
        # for i in self.part
        self.classifiers = nn.ModuleList([nn.Linear(bottleneck_channels, class_num) for _ in range(self.part)])
        self.classifiers.apply(weights_init_classifier)

    def forward(self, x):
        global_features = self.backbone(x)
        outputs = []
        for i in range(self.part):
            part = global_features[:, i * self.part_channels: (i + 1) * self.part_channels, :, :]
            part = self.bottleneck_conv(part).squeeze()
            print(part.shape)
            part = self.classifiers[i](part)
            outputs.append(part)
        return outputs

    def output_features(self, x):
        """extract features of the 6 parts.
            @:return outputs: [batch_size, part * dim_part_feature]
        """

        b, c, h, w = x.size()
        global_features = self.backbone(x)
        outputs = []
        for i in range(self.part):
            part = global_features[:, i * self.part_channels: (i + 1) * self.part_channels, :, :]
            part = self.bottleneck_conv(part)
            outputs.append(part)
        outputs = torch.cat(outputs, dim=2)  # batch_size x channels x 6 x 1
        outputs = outputs.view(b, self.bottleneck_channels, self.part)
        return outputs  # batch_size x bottleneck_channels x num parts

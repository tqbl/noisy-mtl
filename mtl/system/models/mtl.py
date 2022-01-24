from collections import OrderedDict

import torch
import torch.nn as nn

from .vgg import VGGa, conv_block_pooled


class BranchingVGGa(VGGa):
    def __init__(self, n_channels, n_classes, conv_params):
        super().__init__(n_channels, n_classes, conv_params)

        out_channels = conv_params[-1][0]
        self.classifier2 = nn.Linear(out_channels, n_classes)

    def forward(self, x, training=False):
        x = self.bn(x)
        x = self.embedding(x)  # (N, C)
        x1 = self.classifier(x)  # (N, K)

        if not training:
            return x1

        x2 = self.classifier2(x)  # (N, K)
        return x1, x2


class BranchingVGG9a(BranchingVGGa):
    def __init__(self, n_channels, n_classes):
        conv_params = [
            (64, (2, 2), 2),
            (128, (2, 2), 2),
            (256, (2, 2), 2),
            (512, (2, 2), 2),
        ]
        super().__init__(n_channels, n_classes, conv_params)


class BranchingVGG11a(BranchingVGGa):
    def __init__(self, n_channels, n_classes):
        conv_params = [
            (64, (2, 2), 2),
            (128, (2, 2), 2),
            (256, (2, 2), 2),
            (512, (2, 2), 2),
            (512, (2, 2), 2),
        ]
        super().__init__(n_channels, n_classes, conv_params)


class CrossStitchVGGa(nn.Module):
    def __init__(self, n_channels, n_classes, conv_params):
        super().__init__()

        conv_blocks1 = OrderedDict()
        conv_blocks2 = OrderedDict()
        self.stitch = nn.ParameterList()
        for i in range(len(conv_params)):
            out_channels, pool_size, order = conv_params[i]
            in_channels = conv_params[i - 1][0] if i > 0 else n_channels
            conv_blocks1[f'branch1_block{i}'] = conv_block_pooled(
                in_channels, out_channels, pool_size, order=order)
            conv_blocks2[f'branch2_block{i}'] = conv_block_pooled(
                in_channels, out_channels, pool_size, order=order)

            tensor = torch.FloatTensor(out_channels, 2, 2).uniform_(0.1, 0.9)
            self.stitch.append(nn.Parameter(tensor))

        self.bn = nn.BatchNorm2d(n_channels)
        self.conv_blocks1 = nn.Sequential(conv_blocks1)
        self.conv_blocks2 = nn.Sequential(conv_blocks2)
        self.classifier1 = nn.Linear(out_channels, n_classes)
        self.classifier2 = nn.Linear(out_channels, n_classes)

    def forward(self, x, training=False):
        x = self.bn(x)
        x1, x2 = self.embedding(x)  # (N, C)

        x1 = self.classifier1(x1)  # (N, K)
        if not training:
            return x1

        x2 = self.classifier2(x2)  # (N, K)
        return x1, x2

    def embedding(self, x):
        x1 = x2 = x
        for i in range(len(self.conv_blocks1)):
            x1 = self.conv_blocks1[i](x1)
            x2 = self.conv_blocks2[i](x2)
            x1 = torch.einsum('c, ncft -> ncft', self.stitch[i][:, 0, 0], x1) \
                + torch.einsum('c, ncft -> ncft', self.stitch[i][:, 0, 1], x2)
            x2 = torch.einsum('c, ncft -> ncft', self.stitch[i][:, 1, 0], x1) \
                + torch.einsum('c, ncft -> ncft', self.stitch[i][:, 1, 1], x2)

        x1 = x1.amax(dim=(2, 3))  # (N, C)
        x2 = x2.amax(dim=(2, 3))  # (N, C)
        return x1, x2


class CrossStitchVGG9a(CrossStitchVGGa):
    def __init__(self, n_channels, n_classes):
        conv_params = [
            (64, (2, 2), 2),
            (128, (2, 2), 2),
            (256, (2, 2), 2),
            (512, (2, 2), 2),
        ]
        super().__init__(n_channels, n_classes, conv_params)


class CrossStitchVGG11a(CrossStitchVGGa):
    def __init__(self, n_channels, n_classes):
        conv_params = [
            (64, (2, 2), 2),
            (128, (2, 2), 2),
            (256, (2, 2), 2),
            (512, (2, 2), 2),
            (512, (2, 2), 2),
        ]
        super().__init__(n_channels, n_classes, conv_params)

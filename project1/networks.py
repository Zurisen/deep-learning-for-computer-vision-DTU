import os
import numpy as np
import glob
import PIL.Image as Image
from tqdm.notebook import tqdm
import json
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
import torchvision.models as models
from typing import Union, List, Dict, Any, cast


class BasicNetwork(nn.Module):
    def __init__(self, n_filters, out_features=2):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(3, n_filters, kernel_size=4, stride=2, padding=1), 
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.ReLU(),
                        nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.ReLU()
        )
        self.linear = nn.Linear(n_filters*2*8*8, 2)
    def forward(self, x):
        out = self.layers(x)
        out = self.linear(out.view(x.size(0), -1))
        return out


class BasicNetwork_2(nn.Module):
    """ BasicNetwork + Batchnorm """
    def __init__(self, n_filters, out_features=2):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(3, n_filters, kernel_size=4, stride=2, padding=1), 
                        nn.BatchNorm2d(n_filters),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.BatchNorm2d(n_filters),
                        nn.ReLU(),
                        nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(n_filters*2),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.BatchNorm2d(n_filters*2),
                        nn.ReLU()
        )
        self.linear = nn.Linear(n_filters*2*8*8, 2)
        
    def forward(self, x):
        out = self.layers(x)
        out = self.linear(out.view(x.size(0), -1))
        return out


class BasicNetwork_3(nn.Module):
    """ BasicNetwork + Batchnorm + Dropout """
    def __init__(self, n_filters, out_features=2):
        super().__init__()
        self.layers = nn.Sequential(
                        nn.Conv2d(3, n_filters, kernel_size=4, stride=2, padding=1), 
                        nn.BatchNorm2d(n_filters),
                        nn.Dropout(0.3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.BatchNorm2d(n_filters),
                        nn.Dropout(0.3),
                        nn.ReLU(),
                        nn.Conv2d(n_filters, n_filters*2, kernel_size=4, stride=2, padding=1),
                        nn.BatchNorm2d(n_filters*2),
                        nn.Dropout(0.3),
                        nn.ReLU(),
                        nn.MaxPool2d(kernel_size=2, stride=2),
                        nn.BatchNorm2d(n_filters*2),
                        nn.Dropout(0.3),
                        nn.ReLU()
        )
        self.linear = nn.Linear(n_filters*2*8*8, 2)
        
    def forward(self, x):
        out = self.layers(x)
        out = self.linear(out.view(x.size(0), -1))
        return out


class AlexNet(nn.Module):
    def __init__(self, num_classes, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]

def make_layers(batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

class VGG11(nn.Module):
    def __init__(
        self, num_classes, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = make_layers()
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

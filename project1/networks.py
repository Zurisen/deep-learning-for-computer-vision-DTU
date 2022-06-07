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

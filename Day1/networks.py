import os
import tempfile
import numpy as np
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class conv_net(nn.Module):
    def __init__(self, input_ch, output_ch):
        super(conv_net, self).__init__()
        self.input_ch = input_ch 
        self.output_ch = output_ch
        
        self.nn_net = nn.Sequential(
                            nn.Conv2d(input_ch, output_ch*2, 4, 2, 1), # 16x16x8
                            nn.BatchNorm2d(output_ch*2), 
                            nn.ReLU(),

                            nn.Conv2d(output_ch*2, output_ch*4, 4, 2, 1), # 8x8x16
                            nn.BatchNorm2d(output_ch*4), 
                            nn.ReLU(),
              
                            nn.Conv2d(output_ch*4, output_ch*8, 4, 2, 1), # 4x4x32
                            nn.BatchNorm2d(output_ch*8), 
                            nn.ReLU(),
                            
                            nn.Conv2d(output_ch*8, output_ch*16, 4, 2, 1), # 2x2x64
                    )
        self.linear = nn.Sequential(
                        nn.Linear(256, 10),
                        nn.Softmax()
                    )
    def forward(self, x):
        x = self.nn_net(x)
        x = self.linear(x.flatten(start_dim=1))
        return x

class ResNetBlock(nn.Module):
    def __init__(self, n_features):
        super(ResNetBlock, self).__init__()
        self.identity = nn.Identity()
        self.block_layers = nn.Sequential(
            nn.Conv2d(n_features, n_features, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(n_features, n_features, 3, 1, 1)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.block_layers(x)
        out += self.identity(x)
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, n_in, n_features, num_res_blocks=3):
        super(ResNet, self).__init__()
        #First conv layers needs to output the desired number of features.
        conv_layers = [nn.Conv2d(n_in, n_features, kernel_size=3, stride=1, padding=1),
                       nn.ReLU()]
        for i in range(num_res_blocks):
            conv_layers.append(ResNetBlock(n_features))
        self.res_blocks = nn.Sequential(*conv_layers)
        self.fc = nn.Sequential(nn.Linear(32*32*n_features, 2048),
                                nn.ReLU(),
                                nn.Linear(2048, 512),
                                nn.ReLU(),
                                nn.Linear(512,10),
                                nn.Softmax(dim=1))
        
    def forward(self, x):
        x = self.res_blocks(x)
        #reshape x so it becomes flat, except for the first dimension (which is the minibatch)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

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

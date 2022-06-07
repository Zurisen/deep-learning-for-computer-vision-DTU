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
from networks import *
from utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Call the dataset with transformations
size = 128
jitter=0.5
train_transform1 = transforms.Compose([transforms.Resize((size, size)), 
                                      transforms.RandomRotation(degrees=(90)),
                                      transforms.ColorJitter(jitter),
                                    transforms.ToTensor()])
train_transform2 = transforms.Compose([transforms.Resize((size, size)), 
                                      transforms.RandomRotation(degrees=(90)),
                                      transforms.GaussianBlur(3),
                                    transforms.ToTensor()])
train_transform_orig = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.ToTensor()])

test_transform = transforms.Compose([transforms.Resize((size, size)), 
                                    transforms.RandomRotation(degrees=(90)),
                                      transforms.ColorJitter(jitter),
                                      transforms.GaussianBlur(3),
                                    transforms.ToTensor()])
batch_size = 64
trainset_trans1 = Hotdog_NotHotdog(train=True, transform=train_transform1)
trainset_trans2 = Hotdog_NotHotdog(train=True, transform=train_transform2)
trainset_orig = Hotdog_NotHotdog(train=True, transform=train_transform_orig)
all_train=trainset_trans1+trainset_trans2+trainset_orig

train_loader = DataLoader(all_train, batch_size=batch_size, shuffle=True, num_workers=3)
testset = Hotdog_NotHotdog(train=False, transform=test_transform)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=3)

# Our networks
basicnetwork = BasicNetwork(4).to(device)
basicnetwork2 = BasicNetwork_2(4).to(device)
basicnetwork3 = BasicNetwork_3(4).to(device)

# Some state of the art networks
resnet152 = models.resnet152(num_classes=2, pretrained=False).to(device)
alexnet = models.alexnet(num_classes=2, pretrained=False).to(device)
vgg16 = models.vgg16(num_classes=2, pretrained=False).to(device)

# CrossE ntropy Loss function
loss_fun = nn.CrossEntropyLoss()

## SGD and adam optimizers
optimizer_SGD = optim.SGD(basicnetwork.parameters(), lr=0.01)
optimizer_Adam = optim.Adam(basicnetwork.parameters(), lr=0.001)

out_dict = train(resnet152, optimizer_SGD, loss_fun, "ResNet152_SGD", num_epochs=50)
out_dict = train(alexnet, optimizer_SGD, loss_fun, "AlexNet_SGD", num_epochs=50)
out_dict = train(vgg16, optimizer_SGD, loss_fun, "VGG16_SGD", num_epochs=50)


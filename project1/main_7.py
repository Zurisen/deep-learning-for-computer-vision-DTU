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


#We define the training as a function so we can easily re-use it.
def train(model, optimizer, loss_fun, name, num_epochs=30, from_checkpoint=False):
    
    if from_checkpoint:
        model.load_state_dict(torch.load(os.path.join("checkpoints",name)))
    
    out_dict = {'train_acc': [],
              'test_acc': [],
              'train_loss': [],
              'test_loss': [],
              'predictions': [],
              'target': []}
  
    for epoch in tqdm(range(num_epochs), unit='epoch'):
        model.train()
        #For each epoch
        train_correct = 0
        train_loss = []
        model.train()
        for minibatch_no, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data, target = data.to(device), target.to(device)
            #Zero the gradients computed for each weight
            optimizer.zero_grad()
            #Forward pass your image through the network
            output = model(data)
            #Compute the loss
            loss = loss_fun(output, target)
            
            #Backward pass through the network
            loss.backward()
            #Update the weights
            optimizer.step()

            train_loss.append(loss.item())
            #Compute how many were correctly classified
            predicted = output.argmax(1)
            train_correct += (target==predicted).sum().cpu().item()
        #Comput the test accuracy
        test_loss = []
        test_correct = 0
        model.eval()
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                output = model(data)
            test_loss.append(loss_fun(output, target).cpu().item())
            predicted = output.argmax(1)
            test_correct += (target==predicted).sum().cpu().item()
            out_dict["predictions"].append(predicted)
            out_dict["target"].append(target)
        out_dict['train_acc'].append(train_correct/len(all_train))
        out_dict['test_acc'].append(test_correct/len(testset))
        out_dict['train_loss'].append(np.mean(train_loss))
        out_dict['test_loss'].append(np.mean(test_loss))
        print(f"Loss train: {np.mean(train_loss):.3f}\t test: {np.mean(test_loss):.3f}\t",
              f"Accuracy train: {out_dict['train_acc'][-1]*100:.1f}%\t test: {out_dict['test_acc'][-1]*100:.1f}%")
        
        # Saving weights
        torch.save(model.state_dict(), os.path.join("checkpoints",name))
        
        # Saving training/testing info
        info_df = pd.DataFrame()
        info_df['train_acc'] = np.array(out_dict['train_acc']).reshape(-1)
        info_df['test_acc'] = np.array(out_dict['test_acc']).reshape(-1)
        info_df['train_loss'] = np.array(out_dict['train_loss']).reshape(-1)
        info_df['test_loss'] = np.array(out_dict['test_loss']).reshape(-1)
        info_df.to_csv(name+"_info.csv", index=False)
        
        
    return out_dict


# Our networks
basicnetwork = BasicNetwork(4).to(device)
# CrossE ntropy Loss function
loss_fun = nn.CrossEntropyLoss()

## SGD and adam optimizers
optimizer_SGD = optim.SGD(basicnetwork.parameters(), lr=0.01)
optimizer_Adam = optim.Adam(basicnetwork.parameters(), lr=0.001)
out_dict = train(basicnetwork, optimizer_SGD, loss_fun, "BasicNetwork_SGD", num_epochs=50)


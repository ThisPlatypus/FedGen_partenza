import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Subset, DataLoader
from torchvision.datasets import CIFAR100

# create training dataset
def load_datasets(x: int):
    # Download and transform CIFAR-100 (train and test)
    transform = transforms.Compose(
      [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR100("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR100("./dataset", train=False, download=True, transform=transform)

    # We create a tensor that has `True` at an index if the sample belongs to class
    idx0 = torch.tensor(trainset.targets) == x
    idx1 = torch.tensor(trainset.targets) == x+50
    idx = idx0 | idx1
    label_info = {'labels': x + 50, 'counts': sum(idx1)}
    idx0t = torch.tensor(testset.targets) == x
    idx1t = torch.tensor(testset.targets) == x+50
    idxt = idx0t | idx1t
    #We then need to convert this into a list of indices at which we have True.
    train_indices = idx.nonzero().reshape(-1)
    val_indices = idxt.nonzero().reshape(-1)
    #append new subset
    ds_train=torch.utils.data.Subset(trainset,  train_indices)
    ds_val= torch.utils.data.Subset(testset,  val_indices)
    return x, ds_train, ds_val, label_info
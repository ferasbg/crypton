import numbers
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms, datasets
from network import Network

import torch.nn.functional as F
import warnings


class DataSet(data.Dataset):
    def __init__(self, transform=transforms.toTensor()):
        self.transform = transform

    def __getitem__(self, idx):
        img_tensor = self.transform(img)
        return (img_tensor, label)




    def transformInput(self):
        data_transform = transforms.Compose([
            transforms.RandomSizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        cityscapes_dataset = datasets.ImageFolder(root='../../../data/train', transform=data_transform)
        dataset_loader = torch.utils.data.DataLoader(cityscapes_dataset, batch_size=4, shuffle=True, num_workers=4)
        print(dataset_loader)


    def inputToTensor(self):
            """
            Convert input frames into Tensor object.


            """

            pass


def load_model():
    model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
    return model


def check_layers(model):
    """
    precedence property 1: verify network weight state (if member variables are initialized correctly) with getattr()

    """
    raise NotImplementedError

def check_weights():
    """
    precedence property 2: verify network state (if member variables are initialized correctly) with getattr()
    """

    raise NotImplementedError


def preprocess():
    """Pre-process data and pixel-categories. Load data and apply transformations to data, and save to path"""
    raise NotImplementedError

# setup directory
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def save_model(model):
    # save as model state dict
    pass

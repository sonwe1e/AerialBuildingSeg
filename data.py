#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import random
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Dataset(Dataset):
    def __init__(self, transform=None, mode='train'):
        self.transform = transform
        self.mode = mode
        self.label_path = './data/label/'
        self.image_path = './data/image/'
        self.img_list = os.listdir(self.image_path)
        num_img = len(self.img_list)
        random.shuffle(self.img_list)
        if self.mode == 'train':
            self.img_list = self.img_list[:int(num_img*0.8)]
        else:
            self.img_list = self.img_list[int(num_img*0.8):]

    def __getitem__(self, item):
        img_name = self.img_list[item]
        img = cv2.imread(self.image_path + img_name)
        label = cv2.imread(self.label_path + img_name)[...,0] / 255
        if self.transform:
            transformed = self.transform(image=img, mask=label)
            img = transformed['image']
            label = transformed['mask']
        return img, label.long()

    def __len__(self):
        return len(self.img_list)


print('=========Testing Dataset========')
if __name__ == '__main__':
    train_transform = A.Compose([
        # A.Resize(512, 512),
        A.RandomResizedCrop(384, 384),
        A.Flip(),
        A.Normalize(),
        ToTensorV2()
    ])
    test_transform = A.Compose([
        A.Resize(384, 384),
        A.Normalize(),
        ToTensorV2()
    ])
    train = DataLoader(Dataset(transform=train_transform, mode='train'))
    val = DataLoader(Dataset(transform=test_transform, mode='test'))
    for img, label in train:
        print(img.shape, label.shape)

    test_loader = DataLoader(Dataset(mode='test', transform=test_transform), pin_memory=True, batch_size=1, shuffle=False)
    for _, (img, label) in enumerate(test_loader):
        print(img.shape, label.shape)
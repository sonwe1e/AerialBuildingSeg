from unetppp import UNet_Nested
import torch
import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
new_dict = OrderedDict()

model = UNet_Nested()
test_transform = A.Compose([
    A.Resize(height=384, width=384),
    A.Normalize(),
    ToTensorV2()
])

ckpt = torch.load('/home/gdut403/Asonwe1e/AerialBuildingSeg/result/test/1924-0.166.ckpt', map_location='cuda:1')
for k, v in ckpt['state_dict'].items():
    name = k[11:]
    new_dict[name] = v

model.load_state_dict(new_dict)
model.eval()
test_path = './test/'
test_pre_path = './test_pred/'
test_list = os.listdir(test_path)
for test_img in test_list:
    raw_img = cv2.imread(test_path+test_img)
    w, h, _ = raw_img.shape
    img = test_transform(image=raw_img)['image']
    img = img.unsqueeze(0)
    with torch.no_grad():
        pred = model(img)
        pred = torch.argmax(pred, dim=1)
        pred = pred.squeeze(0).cpu().numpy().astype(np.double)
        pred = cv2.resize(pred, (h, w))
        # cv2.imwrite(test_pre_path+test_img, pred*img.cpu().detach().numpy()
        if cv2.imwrite(test_pre_path+test_img, cv2.GaussianBlur(pred, (17, 17), 0)[...,np.newaxis]*raw_img):
            print('success')
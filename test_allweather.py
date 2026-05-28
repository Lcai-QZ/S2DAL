#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-10-24 13:53:35

import os
import cv2
import argparse
import torch
import numpy as np
from pathlib import Path
import time
from utils import batch_PSNR, batch_SSIM
from skimage import img_as_float32, img_as_ubyte
# from networks.derain_net import DerainNet
from basicsr.models.archs import define_network
from basicsr.utils.options import dict2str, parse
from copy import deepcopy
from math import ceil
from torchvision import transforms

import clip

torch.set_default_dtype(torch.float32)

# 设置GPU环境
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser(description='Image Deraining using Restormer')
parser.add_argument('-opt', type=str, default='/media/zyserver/data16t/cailei/project/SAHistoFormer/Allweather/Options/Allweather_Histoformer_Degradation_Aware.yml', help='Path to option YAML file.')
parser.add_argument('--save_path', type=str, default='/media/zyserver/data16t/cailei/project/SAHistoFormer/experiments_allweather', help='save_path')
args = parser.parse_args()
opt = parse(args.opt)

# 加载CLIP模型
inputext = ["Rain degradation with raindrops",
            "Snow degradation with normal snowflakes",
            "Rain degradation with rain lines and normal haze"] # detailed description.

#inputext[0] = "Rain degradation with raindrops"
# inputext = ["Raindrops","Snowflakes"
#             ,"Rain streaks"] # Simple description

# clip_model, _ = clip.load("ViT-B/32", device=0)
clip_model, _ = clip.load(
    "ViT-B/32",
    device=0,
    download_root="/home/cailei/anaconda3/envs/SAHistoFormer/lib/python3.8/site-packages/clip"
)
for param in clip_model.parameters():
    param.requires_grad = False

# 构建网络
print('Loading from {:s}'.format(str(Path('/media/zyserver/data16t/cailei/project/SAHistoFormer/models_save_allweather_text_guided_degration_more_0.5/model_best_ssim_raindrop11.pt'))))
model = define_network(deepcopy(opt['network_g'])).cuda()
# model = DerainNet(n_features=32, n_resblocks=8).cuda()
state_dict = torch.load(str(Path('/media/zyserver/data16t/cailei/project/SAHistoFormer/models_save_allweather_text_guided_degration_more_0.5/model_best_ssim_raindrop11.pt')))
model.load_state_dict(state_dict)
model.eval()

# 加载数据
base_data_path = Path('/media/zyserver/data16t/cailei/data/weather/RainDrop/rain_drop_test')
#rain_types = sorted([x.stem.split('_')[0] for x in base_data_path.glob('*_rain')])
#print(rain_types)
truncate_test = 1  ##
psnr_all_y = []
ssim_all_y = []

rain_dir = Path('/media/zyserver/data16t/cailei/data/weather/RainDrop/rain_drop_test/input')
gt_dir = Path('/media/zyserver/data16t/cailei/data/weather/RainDrop/rain_drop_test/gt')
im_rain_path_list = sorted([x for x in rain_dir.glob('*.png')])
rain_data_list = []
gt_data_list = []

for ii, im_rain_path in enumerate(im_rain_path_list):
    im_rain = img_as_float32(cv2.imread(str(im_rain_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1]))
    im_rain_name = im_rain_path.name
    im_gt_path = gt_dir / im_rain_name.replace('rain', 'clean')
    im_gt = img_as_float32(cv2.imread(str(im_gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1]))

    if im_rain is None or im_gt is None:
        print(f"Failed to read image or ground truth {im_rain_path} or {im_gt_path}. Skipping.")
        continue

    # 确保图像尺寸一致
    _, h, w = im_rain.shape
    if w > 720:
        vaw = w - 720
        im_rain = im_rain[:, :, :-vaw]
        im_gt = im_gt[:, :, :-vaw]
        # im_rain = im_rain[:, :, :-360]
        # im_gt = im_gt[:, :, :-360]
        # im_rain = cv2.resize(im_rain, (720, 480))
        # im_gt = cv2.resize(im_gt, (720, 480))
        if h > 480:
            vah = h - 480
            im_rain = im_rain[:, :-vah, :]
            im_gt = im_gt[:, :-vah, :]
        # im_rain = im_rain[:, :-240, :]
        # im_gt = im_gt[:, :-240, :]
    elif w < 720 or h < 480:
        im_rain = cv2.resize(im_rain, (720, 480))
        im_gt = cv2.resize(im_gt, (720, 480))
    else:
        # im_rain = im_rain[:, :-240, :-360]
        # im_gt = im_gt[:, :-240, :-360]
        im_rain = im_rain
        im_gt = im_gt

    print(im_rain.shape)
    if ii == 0:
        test_data = im_rain[np.newaxis, ]
        test_gt = im_gt[np.newaxis, ]
    else:
        test_data = np.concatenate((test_data, im_rain[np.newaxis,]), axis=0)
        test_gt = np.concatenate((test_gt, im_gt[np.newaxis,]), axis=0)

test_data = torch.from_numpy(img_as_float32(test_data))
test_gt = torch.from_numpy(img_as_float32(test_gt))
current_data_list = [test_data]
num = 0
for kk, current_data in enumerate(current_data_list):
    num_frame = current_data.shape[0]
    test_data_derain = torch.zeros(current_data.shape)
    for ii in range(ceil(num_frame / truncate_test)):
        start_ind = ii * truncate_test
        end_ind = min((ii + 1) * truncate_test, num_frame)
        inputs = current_data[start_ind:end_ind, ].cuda()

        img_id = np.zeros(inputs.shape[0], dtype=int)  ## for raindrop ID
        text_prompt_list = [inputext[idx] for idx in img_id]
        # text_token = clip.tokenize(text_prompt_list).to(1)
        text_token = clip.tokenize(text_prompt_list).cuda()
        # text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)
        text_code = clip_model.encode_text(text_token).to(dtype=torch.float32).cuda()

        with torch.set_grad_enabled(False):
            # out = model(inputs).clamp_(0.0, 1.0).squeeze(0)
            out = model(inputs, text_code).clamp_(0.0, 1.0).squeeze(0)
            test_data_derain[start_ind:end_ind, ] = out.cpu()
            save_out = transforms.ToPILImage()(out.squeeze().cpu())
            save_out.save(os.path.join(args.save_path, 'outputs', str(num)+'_recovery.png'))
            save_gt = transforms.ToPILImage()(test_gt[num].squeeze().cpu())
            save_gt.save(os.path.join(args.save_path, 'gt', str(num)+'_clean.png'))
            num +=1

    if kk == 0:
        psnrm_y = batch_PSNR(test_data_derain, test_gt, ycbcr=True)  ###
        ssimm_y = batch_SSIM(test_data_derain, test_gt, ycbcr=True)  ###

        print('Type:Raindrop, PSNR:{:5.2f}, SSIM:{:6.4f}'.format(psnrm_y, ssimm_y))


#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-09-23 10:23:45

import torch
import cv2
import argparse
import numpy as np
from pathlib import Path
from scipy.io import savemat
from math import ceil
import PIL
import os
import random
import torchvision


parser = argparse.ArgumentParser()
parser.add_argument('--RainDrop_path', type=str, default='/mnt/disk/dataset/weather_diffusion/RainDrop/train',
                    help="Path of the original Rain200H datasets, (default: None)")  ##
parser.add_argument('--train_path', type=str, default='/mnt/disk/dataset/weather_diffusion/RainDrop/train_diffusion',
                    help="Path to save the prepared training datasets, (default: None)") ##
parser.add_argument('--patch_size', type=int, default=64,
                    help="Patch Size, (default: None)")
parser.add_argument('--batch_size', type=int, default=7,
                    help="Batch Size, (default: None)")  ##
parser.add_argument('--patch_n', type=int, default=16,
                    help="Patch Number, (default: None)")  ##
# parser.add_argument('--parse_patches', type=bool, default=True,
#                     help="Parse Patch, (default: None)")

args = parser.parse_args()

overlap = int(args.patch_size/4)
step_size = args.patch_size - overlap

# rain data
rain_floders = Path(args.RainDrop_path) / 'input'  ##
rain_im_list = sorted([x for x in rain_floders.glob('*.png')])   ##
gt_floder = Path(args.RainDrop_path) / 'gt'   ##

transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])



#@staticmethod
def get_params(img, output_size, n):
    w, h = img.size
    th, tw = output_size
    if w == tw and h == th:
        return 0, 0, h, w

    i_list = [random.randint(0, h - th) for _ in range(n)]
    j_list = [random.randint(0, w - tw) for _ in range(n)]
    return i_list, j_list, th, tw

#@staticmethod
def n_random_crops(img, x, y, h, w):
    crops = []
    for i in range(len(x)):
        new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
        crops.append(new_crop)
    return tuple(crops)

for jj, im_path in enumerate(rain_im_list):
    input_img = PIL.Image.open(str(im_path))
    gt_path = gt_floder / (im_path.stem.split('_')[0] + '_clean.png')
    try:
        gt_img = PIL.Image.open(str(gt_path))
    except:
        gt_img = PIL.Image.open(str(gt_path)).convert('RGB')

    if jj==0:
        i, j, h, w = get_params(input_img, (args.patch_size, args.patch_size), args.patch_n)
        input_img = n_random_crops(input_img, i, j, h, w)
        gt_img = n_random_crops(gt_img, i, j, h, w)
        # outputs = [torch.cat([transforms(input_img[i]), transforms(gt_img[i])], dim=0)
        #                for i in range(args.patch_n)]
        input = [transforms(input_img[i]) for i in range(args.patch_n)]
        gt = [transforms(gt_img[i]) for i in range(args.patch_n)]

        input_st = torch.stack(input, dim=0)
        gt_st = torch.stack(gt, dim=0)

        input_num = input_st.numpy()
        gt_num = gt_st.numpy()

        rain_data_temp = input_num[np.newaxis, ] # 16 X 3 X 1 X 64 X 64
        gt_data_temp = gt_num[np.newaxis, ] # 16 X 3 X 1 X 64 X 64

    else:
        i, j, h, w = get_params(input_img, (args.patch_size, args.patch_size), args.patch_n)
        input_img = n_random_crops(input_img, i, j, h, w)
        gt_img = n_random_crops(gt_img, i, j, h, w)
        # outputs = [torch.cat([transforms(input_img[i]), transforms(gt_img[i])], dim=0)
        #                for i in range(args.patch_n)]
        input = [transforms(input_img[i]) for i in range(args.patch_n)]
        gt = [transforms(gt_img[i]) for i in range(args.patch_n)]

        input_st = torch.stack(input, dim=0)
        gt_st = torch.stack(gt, dim=0)

        input_num = input_st.numpy()
        gt_num = gt_st.numpy()

        rain_data_temp_1 = input_num[np.newaxis, ]  # 16 X 3 X 1 X 64 X 64
        gt_data_temp_1 = gt_num[np.newaxis, ]  # 16 X 3 X 1 X 64 X 64

        rain_data_temp = np.concatenate((rain_data_temp, rain_data_temp_1), axis=0)
        gt_data_temp = np.concatenate((gt_data_temp, rain_data_temp_1), axis=0)


    # im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
    # gt_path = gt_floder / (im_path.stem.split('_')[0] + '_clean.png')
    # gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
    # if jj == 0:
    #     iter_patch = 0
    #     rain_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
    #     gt_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
    #     c, num_frame, h, w = rain_data_temp.shape
    #     inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
    #     inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
    #     num_patch = len(inds_h) * len(inds_w)
    #     rain_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
    #     gt_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
    #     # iter_patch = 0
    #     for hh in inds_h:
    #         for ww in inds_w:
    #             rain_data[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #             gt_data[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #             iter_patch += 1
    #     num_frame += 1
    # else:
    #     iter_patch = 0
    #     rain_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
    #     gt_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
    #     c, _, h, w = rain_data_temp.shape
    #     inds_h_1 = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
    #     inds_w_1 = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
    #     # num_patch_1 = len(inds_h_1) * len(inds_w_1)
    #     if inds_h_1 <= inds_h and inds_w_1 <= inds_w:
    #         num_patch = len(inds_h_1) * len(inds_w_1)
    #         inds_h = inds_h_1
    #         inds_w = inds_w_1
    #         rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
    #         gt_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
    #         for hh in inds_h:
    #             for ww in inds_w:
    #                 rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #                 gt_data_1[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #                 iter_patch += 1
    #         rain_data = np.concatenate((rain_data, rain_data_1), axis=2)  # c x num_frame x h x w
    #         gt_data = np.concatenate((gt_data[:iter_patch, :, :, :], gt_data_1), axis=2)  # c x num_frame x h x w
    #         # rain_data = np.concatenate(rain_data[:iter_patch, :, :, :], rain_data_1, axis=2)  # c x num_frame x h x w
    #     elif inds_h_1 <= inds_h and inds_w_1 >= inds_w:
    #         inds_h = inds_h_1
    #         num_patch = len(inds_h) * len(inds_w)
    #         rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
    #         gt_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
    #         for hh in inds_h:
    #             for ww in inds_w:
    #                 rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #                 gt_data_1[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #                 iter_patch += 1
    #         rain_data = np.concatenate((rain_data[:iter_patch, :, :, :], rain_data_1), axis=2)  # c x num_frame x h x w
    #         gt_data = np.concatenate((gt_data[:iter_patch, :, :, :], gt_data_1), axis=2)  # c x num_frame x h x w
    #     elif inds_h_1 >= inds_h and inds_w_1 <= inds_w:
    #         inds_w = inds_w_1
    #         num_patch = len(inds_h) * len(inds_w)
    #         rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
    #         gt_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
    #         for hh in inds_h:
    #             for ww in inds_w:
    #                 rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #                 gt_data_1[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #                 iter_patch += 1
    #         rain_data = np.concatenate((rain_data[:iter_patch, :, :, :], rain_data_1), axis=2)  # c x num_frame x h x w
    #         gt_data = np.concatenate((gt_data[:iter_patch, :, :, :], gt_data_1), axis=2)  # c x num_frame x h x w
    #     else:
    #         rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
    #         gt_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
    #         for hh in inds_h:
    #             for ww in inds_w:
    #                 rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #                 gt_data_1[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
    #                 iter_patch += 1
    #         rain_data = np.concatenate((rain_data, rain_data_1), axis=2)  # c x num_frame x h x w
    #         gt_data = np.concatenate((gt_data, gt_data_1), axis=2)  # c x num_frame x h x w

assert rain_data_temp.shape == gt_data_temp.shape

img_num = len(rain_im_list)


for nn in range(img_num):
    save_patch = Path(args.train_path) / ('t' + str(nn) + '_rain_' + '.mat')
    # save_patch = Path(args.train_path) / ('t'+str(1)+'_rain_'+str(kk+1)+'_'+str(ss+1)+'.mat')
    #     if save_patch.exists():
    #         save_patch.unlink()
    if nn == 0:
        # single_rain_data_batch = rain_data_temp[:,:,:nn+1,:]
        single_rain_data_batch = rain_data_temp[:nn + 1, :, :, :]
        # savemat(str(save_patch), {'rain_data':single_rain_data_batch.squeeze(2)})
        savemat(str(save_patch), {'rain_data': single_rain_data_batch})
    else:
        # single_rain_data_batch = rain_data_temp[:,:,nn:nn+1,:]
        single_rain_data_batch = rain_data_temp[nn:nn + 1, :, :, :]
        # savemat(str(save_patch), {'rain_data':single_rain_data_batch.squeeze(2)})
        savemat(str(save_patch), {'rain_data': single_rain_data_batch})

# for ss in range(ceil(num_patch / args.batch_size)):
#     start = ss * args.batch_size
#     end = min((ss + 1) * args.batch_size, num_patch)
#     rain_data_batch = rain_data[start:end, ]
#     for nn in range(img_num):
#         save_patch = Path(args.train_path) / ('t' + str(nn) + '_rain_' + str(ss + 1) + '.mat')
#     # save_patch = Path(args.train_path) / ('t'+str(1)+'_rain_'+str(kk+1)+'_'+str(ss+1)+'.mat')
#     #     if save_patch.exists():
#     #         save_patch.unlink()
#         if nn == 0:
#             single_rain_data_batch = rain_data_batch[:,:,:nn+1,:]
#             savemat(str(save_patch), {'rain_data':single_rain_data_batch.squeeze(2)})
#         else:
#             single_rain_data_batch = rain_data_batch[:,:,nn:nn+1,:]
#             savemat(str(save_patch), {'rain_data':single_rain_data_batch.squeeze(2)})



for nn in range(img_num):
    save_patch = Path(args.train_path) / ('t' + str(nn) + '_gt_' + '.mat')
    # if save_patch.exists():
    #     save_patch.unlink()
    # savemat(str(save_patch), {'gt_data':gt_data_batch})
    if nn == 0:
        single_gt_data_batch = gt_data_temp[:nn+1, :,:,:]
        # single_gt_data_batch = gt_data_temp[:, :, :nn + 1, :]
        savemat(str(save_patch), {'gt_data': single_gt_data_batch})
        # savemat(str(save_patch), {'gt_data': single_gt_data_batch.squeeze(2)})
    else:
        single_gt_data_batch = gt_data_temp[nn:nn+1, :, :, :]
        # single_gt_data_batch = gt_data_temp[:, :, nn:nn + 1, :]
        savemat(str(save_patch), {'gt_data': single_gt_data_batch})
        # savemat(str(save_patch), {'gt_data': single_gt_data_batch.squeeze(2)})

# for kk in range(ceil(num_patch / args.batch_size)):
#     start = kk * args.batch_size
#     end = min((kk + 1) * args.batch_size, num_patch)
#     gt_data_batch = gt_data[start:end, ]
#     for nn in range(img_num):
#         save_patch = Path(args.train_path) / ('t' + str(nn) + '_gt_' + str(kk + 1) + '.mat')
#         # if save_patch.exists():
#         #     save_patch.unlink()
#         # savemat(str(save_patch), {'gt_data':gt_data_batch})
#         if nn == 0:
#             single_gt_data_batch = gt_data_batch[:,:,:nn+1,:]
#             savemat(str(save_patch), {'gt_data': single_gt_data_batch.squeeze(2)})
#         else:
#             single_gt_data_batch = gt_data_batch[:, :, nn:nn+1, :]
#             savemat(str(save_patch), {'gt_data': single_gt_data_batch.squeeze(2)})
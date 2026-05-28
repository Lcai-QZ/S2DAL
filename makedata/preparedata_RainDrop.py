#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-09-23 10:23:45

import cv2
import argparse
import numpy as np
from pathlib import Path
from scipy.io import savemat
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument('--RainDrop_path', type=str, default='/media/zyserver/data16t/cailei/data/weather/RainDrop/train',
                    help="Path of the original Rain200H datasets, (default: None)")  ##
parser.add_argument('--train_path', type=str, default='/media/zyserver/data16t/cailei/data/weather/RainDrop/train_mat_128_4',
                    help="Path to save the prepared training datasets, (default: None)") ##
parser.add_argument('--patch_size', type=int, default=128,
                    help="Patch Size, (default: None)")  ##
parser.add_argument('--batch_size', type=int, default=4,
                    help="Batch Size, (default: None)")  ##

args = parser.parse_args()

overlap = int(args.patch_size/4)
step_size = args.patch_size - overlap

# rain data
rain_floders = Path(args.RainDrop_path) / 'input'  ##
rain_im_list = sorted([x for x in rain_floders.glob('*.png')])   ##
gt_floder = Path(args.RainDrop_path) / 'gt'   ##
for jj, im_path in enumerate(rain_im_list):
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
    gt_path = gt_floder / (im_path.stem.split('_')[0] + '_clean.png')
    gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
    if jj == 0:
        iter_patch = 0
        rain_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        gt_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
        c, num_frame, h, w = rain_data_temp.shape
        inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        num_patch = len(inds_h) * len(inds_w)
        rain_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        gt_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        # iter_patch = 0
        for hh in inds_h:
            for ww in inds_w:
                rain_data[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                gt_data[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                iter_patch += 1
        num_frame += 1
    else:
        iter_patch = 0
        rain_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        gt_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
        c, _, h, w = rain_data_temp.shape
        inds_h_1 = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w_1 = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        # num_patch_1 = len(inds_h_1) * len(inds_w_1)
        if inds_h_1 <= inds_h and inds_w_1 <= inds_w:
            num_patch = len(inds_h_1) * len(inds_w_1)
            inds_h = inds_h_1
            inds_w = inds_w_1
            rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_data_1[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rain_data = np.concatenate((rain_data, rain_data_1), axis=2)  # c x num_frame x h x w
            gt_data = np.concatenate((gt_data[:iter_patch, :, :, :], gt_data_1), axis=2)  # c x num_frame x h x w
            # rain_data = np.concatenate(rain_data[:iter_patch, :, :, :], rain_data_1, axis=2)  # c x num_frame x h x w
        elif inds_h_1 <= inds_h and inds_w_1 >= inds_w:
            inds_h = inds_h_1
            num_patch = len(inds_h) * len(inds_w)
            rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_data_1[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rain_data = np.concatenate((rain_data[:iter_patch, :, :, :], rain_data_1), axis=2)  # c x num_frame x h x w
            gt_data = np.concatenate((gt_data[:iter_patch, :, :, :], gt_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 >= inds_h and inds_w_1 <= inds_w:
            inds_w = inds_w_1
            num_patch = len(inds_h) * len(inds_w)
            rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_data_1[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rain_data = np.concatenate((rain_data[:iter_patch, :, :, :], rain_data_1), axis=2)  # c x num_frame x h x w
            gt_data = np.concatenate((gt_data[:iter_patch, :, :, :], gt_data_1), axis=2)  # c x num_frame x h x w
        else:
            rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_data_1[iter_patch,] = gt_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rain_data = np.concatenate((rain_data, rain_data_1), axis=2)  # c x num_frame x h x w
            gt_data = np.concatenate((gt_data, gt_data_1), axis=2)  # c x num_frame x h x w

assert gt_data.shape == rain_data.shape

img_num = len(rain_im_list)

for ss in range(ceil(num_patch / args.batch_size)):
    start = ss * args.batch_size
    end = min((ss + 1) * args.batch_size, num_patch)
    rain_data_batch = rain_data[start:end, ]
    for nn in range(img_num):
        save_patch = Path(args.train_path) / ('t' + str(nn) + '_rain_' + str(ss + 1) + '.mat')
    # save_patch = Path(args.train_path) / ('t'+str(1)+'_rain_'+str(kk+1)+'_'+str(ss+1)+'.mat')
    #     if save_patch.exists():
    #         save_patch.unlink()
        if nn == 0:
            single_rain_data_batch = rain_data_batch[:,:,:nn+1,:]
            savemat(str(save_patch), {'rain_data':single_rain_data_batch.squeeze(2)})
        else:
            single_rain_data_batch = rain_data_batch[:,:,nn:nn+1,:]
            savemat(str(save_patch), {'rain_data':single_rain_data_batch.squeeze(2)})


for kk in range(ceil(num_patch / args.batch_size)):
    start = kk * args.batch_size
    end = min((kk + 1) * args.batch_size, num_patch)
    gt_data_batch = gt_data[start:end, ]
    for nn in range(img_num):
        save_patch = Path(args.train_path) / ('t' + str(nn) + '_gt_' + str(kk + 1) + '.mat')
        # if save_patch.exists():
        #     save_patch.unlink()
        # savemat(str(save_patch), {'gt_data':gt_data_batch})
        if nn == 0:
            single_gt_data_batch = gt_data_batch[:,:,:nn+1,:]
            savemat(str(save_patch), {'gt_data': single_gt_data_batch.squeeze(2)})
        else:
            single_gt_data_batch = gt_data_batch[:, :, nn:nn+1, :]
            savemat(str(save_patch), {'gt_data': single_gt_data_batch.squeeze(2)})


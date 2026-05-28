#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import argparse
import numpy as np
from pathlib import Path
from scipy.io import savemat
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument('--ntu_path_semi', type=str, default='/media/zyserver/data16t/cailei/data/weather/RainDrop/test_b',
                    help="Path to save the original NTURain datasets, (default: None)")
parser.add_argument('--train_path', type=str, default='/media/zyserver/data16t/cailei/data/weather/RainDrop/train_semi_mat_128_4/',
                    help="Path to save the prepared training datasets, (default: None)")
parser.add_argument('--patch_size', type=int, default=128,
                    help="Patch size for cropping")
parser.add_argument('--batch_size', type=int, default=4,
                    help="Batch size for saving data") ##
args = parser.parse_args()

overlap = int(args.patch_size/4)
step_size = args.patch_size - overlap

rain_dir = Path(args.ntu_path_semi) / 'data'
im_rain_path_list = sorted([x for x in rain_dir.glob('*.jpg')])
for ii, im_rain_path in enumerate(im_rain_path_list):
    im_rain = cv2.imread(str(im_rain_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    if ii == 0:
        iter_patch = 0
        rain_data_temp = im_rain[:, np.newaxis, ]  # c x 1 x h x w
        c, num_frame, h, w = rain_data_temp.shape
        inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        num_patch = len(inds_h) * len(inds_w)
        rain_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        # iter_patch = 0
        for hh in inds_h:
            for ww in inds_w:
                rain_data[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                iter_patch += 1
        num_frame += 1
    else:
        iter_patch = 0
        rain_data_temp = im_rain[:, np.newaxis, ]  # c x 1 x h x w
        c, _, h, w = rain_data_temp.shape
        inds_h_1 = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w_1 = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        if inds_h_1 <= inds_h and inds_w_1 <= inds_w:
            num_patch = len(inds_h_1) * len(inds_w_1)
            inds_h = inds_h_1
            inds_w = inds_w_1
            rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rain_data = np.concatenate((rain_data[:iter_patch, :, :, :], rain_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 <= inds_h and inds_w_1 >= inds_w:
            inds_h = inds_h_1
            num_patch = len(inds_h) * len(inds_w)
            rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rain_data = np.concatenate((rain_data[:iter_patch, :, :, :], rain_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 >= inds_h and inds_w_1 <= inds_w:
            inds_w = inds_w_1
            num_patch = len(inds_h) * len(inds_w)
            rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rain_data = np.concatenate((rain_data[:iter_patch, :, :, :], rain_data_1), axis=2)  # c x num_frame x h x w
        else:
            rain_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rain_data_1[iter_patch,] = rain_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rain_data = np.concatenate((rain_data, rain_data_1), axis=2)  # c x num_frame x h x w

# assert gt_data.shape == rain_data.shape
img_num = len(im_rain_path_list)

# assert gt_data.shape == rain_data.shape
for kk in range(ceil(num_patch / args.batch_size)):
    start = kk * args.batch_size
    end = min((kk + 1) * args.batch_size, num_patch)
    rain_data_batch = rain_data[start:end, ]
    for nn in range(img_num):
        # save_patch = Path(args.train_path) / ('real' + '_rain_' + str(kk + 1) + '.mat')
        save_patch = Path(args.train_path) / ('real' + str(nn) + '_rain_' + str(kk + 1) + '.mat')
    # if save_patch.exists():
    #     save_patch.unlink()
        if nn == 0:
            single_rain_data_batch = rain_data_batch[:,:,:nn+1,:]
            savemat(str(save_patch), {'rain_data': single_rain_data_batch.squeeze(2)})
        else:
            single_rain_data_batch = rain_data_batch[:,:,nn:nn+1,:]
            savemat(str(save_patch), {'rain_data':single_rain_data_batch.squeeze(2)})

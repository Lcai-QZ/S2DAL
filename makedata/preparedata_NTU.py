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
parser.add_argument('--ntu_path', type=str, default='/mnt/disk/dataset/RainDrop/raindrop_train_data', 
                                help="Path of the original NTURain datasets, (default: None)")   ##
parser.add_argument('--train_path', type=str, default='/mnt/disk/dataset/RainDrop/train_256/',
                                help="Path to save the prepared training datasets, (default: None)")   ## 
parser.add_argument('--patch_size', type=int, default=64,
                                help="Path to save the prepared training datasets, (default: None)")   ## default=64
parser.add_argument('--batch_size', type=int, default=10,
                                help="Path to save the prepared training datasets, (default: None)")   ## default=12
args = parser.parse_args()

overlap = int(args.patch_size/4)
step_size = args.patch_size - overlap

# for ii in range(1, 9):    
for ii in range(1, 2):     
    print('Scene {:d}'.format(ii))
    # ground truth data
    gt_floder = Path(args.ntu_path) / Path('t'+str(ii)+'_GT')
    gt_im_list = sorted([x for x in gt_floder.glob('*.png')])
    for jj, im_path in enumerate(gt_im_list):
        im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
        print(im_path)
        if jj == 0:
            gt_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        else:
            gt_data_temp = np.concatenate((gt_data_temp, im[:, np.newaxis,]), axis=1) # c x num_frame x h x w
    # for jj, im_path in enumerate(gt_im_list):
    #     im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)
    #     if im is None:
    #         print(f"Error loading image: {im_path}")
    #         continue
    #     if im.shape[0] != args.patch_size or im.shape[1] != args.patch_size:
    #             im = cv2.resize(im, (args.patch_size, args.patch_size))
    #     im = im.transpose([2,0,1])
    #     if jj == 0:
    #         gt_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
    #     else:
    #         gt_data_temp = np.concatenate((gt_data_temp, im[:, np.newaxis,]), axis=1) # c x num_frame x h x w
    # crop groundtruth into patch
    c, num_frame, h, w = gt_data_temp.shape
    inds_h = list(range(0, h-args.patch_size, step_size)) + [h-args.patch_size,]
    inds_w = list(range(0, w-args.patch_size, step_size)) + [w-args.patch_size,]
    num_patch = len(inds_h) * len(inds_w)
    gt_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
    iter_patch = 0
    for hh in inds_h:
        for ww in inds_w:
            gt_data[iter_patch, ] = gt_data_temp[:, :, hh:hh+args.patch_size, ww:ww+args.patch_size]
            iter_patch += 1

    for kk in range(ceil(num_patch / args.batch_size)):
        start = kk * args.batch_size
        end = min((kk+1)*args.batch_size, num_patch)
        gt_data_batch = gt_data[start:end,]
        save_path = Path(args.train_path) / ('t'+str(ii)+'_gt_'+str(kk+1)+'.mat')
        if save_path.exists():
            save_path.unlink()
        savemat(str(save_path), {'gt_data':gt_data_batch})

    # rain data
    rain_floders = sorted([x for x in Path(args.ntu_path).glob('t'+str(ii)+'_Rain_*')])
    for kk, current_floder in enumerate(rain_floders):
        print('    Rain type: {:d}'.format(kk+1))
        rain_im_list = sorted([x for x in current_floder.glob('*.png')])
        for jj, im_path in enumerate(rain_im_list):
            im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
            print(im_path)
            if jj == 0:
                rain_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
            else:
                rain_data_temp = np.concatenate((rain_data_temp, im[:, np.newaxis,]), axis=1) # c x num_frame x h x w
        # for jj, im_path in enumerate(rain_im_list):
        #     ##########################
        #     im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)
        #     if im is None:
        #         print(f"Error loading image: {im_path}")
        #         continue
        #     if im.shape[0] != args.patch_size or im.shape[1] != args.patch_size:
        #         im = cv2.resize(im, (args.patch_size, args.patch_size))
        #     im = im.transpose([2,0,1])
        #     ##########################
        #     if jj == 0:
        #         rain_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        #     else:
        #         rain_data_temp = np.concatenate((rain_data_temp, im[:, np.newaxis,]), axis=1) # c x num_frame x h x w
        assert gt_data_temp.shape == rain_data_temp.shape
        # crop rain data into patch
        rain_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        iter_patch = 0
        for hh in inds_h:
            for ww in inds_w:
                rain_data[iter_patch, ] = rain_data_temp[:, :, hh:hh+args.patch_size, ww:ww+args.patch_size]
                iter_patch += 1

        for ss in range(ceil(num_patch / args.batch_size)):
            start = ss * args.batch_size
            end = min((ss+1)*args.batch_size, num_patch)
            rain_data_batch = rain_data[start:end,]
            save_path = Path(args.train_path) / ('t'+str(ii)+'_rain_'+str(kk+1)+'_'+str(ss+1)+'.mat')
            if save_path.exists():
                save_path.unlink()
            savemat(str(save_path), {'rain_data':rain_data_batch})


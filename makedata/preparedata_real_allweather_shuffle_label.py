import cv2
import argparse
import os
import numpy as np
from pathlib import Path
from scipy.io import savemat
from math import ceil
import random
from weather_frame_tracker import make_frame_labels_equal_blocks, apply_same_shuffle_1d, FrameCategoryTracker


parser = argparse.ArgumentParser()
parser.add_argument('--allweather_path', type=str, default='/media/zyserver/data16t/cailei/data/weather_real/train_rf/synthetic',
                    help="Path of the original weather datasets, (default: None)")  ##
parser.add_argument('--train_path', type=str, default='/media/zyserver/data16t/cailei/data/weather_real/allweather_train_real_mat_128_label_shuffle',
                    help="Path to save the prepared training datasets, (default: None)") ##
parser.add_argument('--patch_size', type=int, default=128,
                    help="Patch Size, (default: None)")  ##
parser.add_argument('--batch_size', type=int, default=12,
                    help="Batch Size, (default: None)")  ## setting batch_size to 3 or 6

args = parser.parse_args()

overlap = int(args.patch_size/4)
step_size = args.patch_size - overlap

# weather data
snow_gt_floders = Path(args.allweather_path) / 'Snow' / 'gt'  ##
snow_floders = Path(args.allweather_path) / 'Snow' / 'input'  ##
snow_im_list = sorted([x for x in snow_floders.glob('*')])   ##
rain_gt_floders = Path(args.allweather_path) / 'Rain' / 'gt'  ##
rain_floders = Path(args.allweather_path) / 'Rain' / 'input'  ##
rain_im_list = sorted([x for x in rain_floders.glob('*')])   ##
haze_gt_floders = Path(args.allweather_path) / 'Haze' / 'gt'  ##
haze_floders = Path(args.allweather_path) / 'Haze' / 'input'  ##
haze_im_list = sorted([x for x in haze_floders.glob('*')])   ##

#--------------------for raindrop----------------------------#
for jj, im_path in enumerate(rain_im_list):
    print('rain:', jj)
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
    # gt_path = gt_floder / (im_path.stem.split('.')[0] + '.png'
    filename = os.path.basename(im_path)
    gt_path = rain_gt_floders / filename
    gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    if jj == 0:
        iter_patch = 0
        raindrop_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        gt_raindrop_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
        c, num_frame, h, w = raindrop_data_temp.shape
        inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        num_patch = len(inds_h) * len(inds_w)
        raindrop_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        gt_raindrop_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        # iter_patch = 0
        for hh in inds_h:
            for ww in inds_w:
                raindrop_data[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                gt_raindrop_data[iter_patch,] = gt_raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                iter_patch += 1
        num_frame +=1
    else:
        iter_patch = 0
        raindrop_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        gt_raindrop_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
        c, _, h, w = raindrop_data_temp.shape
        inds_h_1 = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w_1 = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        # num_patch_1 = len(inds_h_1) * len(inds_w_1)
        if inds_h_1 <= inds_h and inds_w_1 <= inds_w:
            num_patch = len(inds_h_1) * len(inds_w_1)
            inds_h = inds_h_1
            inds_w = inds_w_1
            raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    raindrop_data_1[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_raindrop_data_1[iter_patch,] = gt_raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            raindrop_data = np.concatenate((raindrop_data, raindrop_data_1), axis=2)
            gt_raindrop_data = np.concatenate((gt_raindrop_data[:iter_patch, :, :, :], gt_raindrop_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 <= inds_h and inds_w_1 >= inds_w:
            inds_h = inds_h_1
            num_patch = len(inds_h) * len(inds_w)
            raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    raindrop_data_1[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_raindrop_data_1[iter_patch,] = gt_raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            raindrop_data = np.concatenate((raindrop_data[:iter_patch, :, :, :], raindrop_data_1), axis=2)  # c x num_frame x h x w
            gt_raindrop_data = np.concatenate((gt_raindrop_data[:iter_patch, :, :, :], gt_raindrop_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 >= inds_h and inds_w_1 <= inds_w:
            inds_w = inds_w_1
            num_patch = len(inds_h) * len(inds_w)
            raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                   raindrop_data_1[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                   gt_raindrop_data_1[iter_patch,] = gt_raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                   iter_patch += 1
            raindrop_data = np.concatenate((raindrop_data[:iter_patch, :, :, :], raindrop_data_1), axis=2)
            gt_raindrop_data = np.concatenate((gt_raindrop_data[:iter_patch, :, :, :], gt_raindrop_data_1), axis=2)
        else:
            raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    raindrop_data_1[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_raindrop_data_1[iter_patch,] = gt_raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            raindrop_data = np.concatenate((raindrop_data, raindrop_data_1), axis=2)  # c x num_frame x h x w
            gt_raindrop_data = np.concatenate((gt_raindrop_data, gt_raindrop_data_1), axis=2)  # c x num_frame x h x w

assert gt_raindrop_data.shape == raindrop_data.shape
# raindrop_img_num = len(raindrop_im_list)


#--------------------for snow----------------------------#
for jj, im_path in enumerate(snow_im_list):
    print('snow:', jj)
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    # gt_path = gt_floder / (im_path.stem.split('.')[0] + '.jpg')
    filename = os.path.basename(im_path)
    gt_path = snow_gt_floders / filename
    gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    if jj == 0:
        iter_patch = 0
        snow_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        gt_snow_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
        c, num_frame, h, w = snow_data_temp.shape
        inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        num_patch = len(inds_h) * len(inds_w)
        snow_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        gt_snow_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        # iter_patch = 0
        for hh in inds_h:
            for ww in inds_w:
                snow_data[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                gt_snow_data[iter_patch,] = gt_snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                iter_patch += 1
        num_frame += 1
    else:
        iter_patch = 0
        snow_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        gt_snow_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
        c, _, h, w = snow_data_temp.shape
        inds_h_1 = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w_1 = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        # num_patch_1 = len(inds_h_1) * len(inds_w_1)
        if inds_h_1 <= inds_h and inds_w_1 <= inds_w:
            num_patch = len(inds_h_1) * len(inds_w_1)
            inds_h = inds_h_1
            inds_w = inds_w_1
            snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    snow_data_1[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_snow_data_1[iter_patch,] = gt_snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            snow_data = np.concatenate((snow_data, snow_data_1), axis=2)
            gt_snow_data = np.concatenate((gt_snow_data[:iter_patch, :, :, :], gt_snow_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 <= inds_h and inds_w_1 >= inds_w:
            inds_h = inds_h_1
            num_patch = len(inds_h) * len(inds_w)
            snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    snow_data_1[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_snow_data_1[iter_patch,] = gt_snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            snow_data = np.concatenate((snow_data[:iter_patch, :, :, :], snow_data_1), axis=2)  # c x num_frame x h x w
            gt_snow_data = np.concatenate((gt_snow_data[:iter_patch, :, :, :], gt_snow_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 >= inds_h and inds_w_1 <= inds_w:
            inds_w = inds_w_1
            num_patch = len(inds_h) * len(inds_w)
            snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    snow_data_1[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_snow_data_1[iter_patch,] = gt_snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            snow_data = np.concatenate((snow_data[:iter_patch, :, :, :], snow_data_1), axis=2)
            gt_snow_data = np.concatenate((gt_snow_data[:iter_patch, :, :, :], gt_snow_data_1), axis=2)
        else:
            snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    snow_data_1[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_snow_data_1[iter_patch,] = gt_snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            snow_data = np.concatenate((snow_data, snow_data_1), axis=2)  # c x num_frame x h x w
            gt_snow_data = np.concatenate((gt_snow_data, gt_snow_data_1), axis=2)  # c x num_frame x h x w

assert gt_snow_data.shape == snow_data.shape
snow_img_num = len(snow_im_list)

#--------------------for haze----------------------------#
for jj, im_path in enumerate(haze_im_list):
    print('haze:', jj)
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    filename = os.path.basename(im_path)
    # gt_path = gt_floder / (im_path.stem.split('.')[0] + '.png')
    gt_path = haze_gt_floders / filename
    gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    if jj == 0:
        iter_patch = 0
        rainhaze_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        gt_rainhaze_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
        c, num_frame, h, w = rainhaze_data_temp.shape
        inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        num_patch = len(inds_h) * len(inds_w)
        rainhaze_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        gt_rainhaze_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        # iter_patch = 0
        for hh in inds_h:
            for ww in inds_w:
                rainhaze_data[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                gt_rainhaze_data[iter_patch,] = gt_rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                iter_patch += 1
        num_frame +=1
    else:
        iter_patch = 0
        rainhaze_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        gt_rainhaze_data_temp = gt[:, np.newaxis, ]  # c x 1 x h x w
        c, _, h, w = rainhaze_data_temp.shape
        inds_h_1 = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w_1 = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        # num_patch_1 = len(inds_h_1) * len(inds_w_1)
        if inds_h_1 <= inds_h and inds_w_1 <= inds_w:
            num_patch = len(inds_h_1) * len(inds_w_1)
            inds_h = inds_h_1
            inds_w = inds_w_1
            rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rainhaze_data_1[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_rainhaze_data_1[iter_patch,] = gt_rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rainhaze_data = np.concatenate((rainhaze_data, rainhaze_data_1), axis=2)
            gt_rainhaze_data = np.concatenate((gt_rainhaze_data[:iter_patch, :, :, :], gt_rainhaze_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 <= inds_h and inds_w_1 >= inds_w:
            inds_h = inds_h_1
            num_patch = len(inds_h) * len(inds_w)
            rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rainhaze_data_1[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_rainhaze_data_1[iter_patch,] = gt_rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rainhaze_data = np.concatenate((rainhaze_data[:iter_patch, :, :, :], rainhaze_data_1), axis=2)  # c x num_frame x h x w
            gt_rainhaze_data = np.concatenate((gt_rainhaze_data[:iter_patch, :, :, :], gt_rainhaze_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 >= inds_h and inds_w_1 <= inds_w:
            inds_w = inds_w_1
            num_patch = len(inds_h) * len(inds_w)
            rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rainhaze_data_1[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_rainhaze_data_1[iter_patch,] = gt_rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rainhaze_data = np.concatenate((rainhaze_data[:iter_patch, :, :, :], rainhaze_data_1), axis=2)
            gt_rainhaze_data = np.concatenate((gt_rainhaze_data[:iter_patch, :, :, :], gt_rainhaze_data_1), axis=2)
        else:
            rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            gt_rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rainhaze_data_1[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    gt_rainhaze_data_1[iter_patch,] = gt_rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rainhaze_data = np.concatenate((rainhaze_data, rainhaze_data_1), axis=2)  # c x num_frame x h x w
            gt_rainhaze_data = np.concatenate((gt_rainhaze_data, gt_rainhaze_data_1), axis=2)  # c x num_frame x h x w

assert gt_rainhaze_data.shape == rainhaze_data.shape
# rainhaze_img_num = len(rainhaze_im_list)


#------------------ for snow ------------------#
for nn in range(ceil(35 / 8)):
    start = nn * 8
    end = min((nn + 1) * 8, 40)
    if start == 0:
        snow_data_0 = snow_data[start:end, :, :, :]
        gt_snow_data_0 = gt_snow_data[start:end, :, :, :]
    elif start == 8:
        snow_data_1 = snow_data[start:end, :, :, :]
        snow_data_rerange = np.concatenate((snow_data_0, snow_data_1), axis=2)
        gt_snow_data_1 = gt_snow_data[start:end, :, :, :]
        gt_snow_data_rerange = np.concatenate((gt_snow_data_0, gt_snow_data_1), axis=2)
    else:
        snow_data_1 = snow_data[start:end, :, :, :]
        snow_data_rerange = np.concatenate((snow_data_rerange, snow_data_1), axis=2)
        gt_snow_data_1 = gt_snow_data[start:end, :, :, :]
        gt_snow_data_rerange = np.concatenate((gt_snow_data_rerange, gt_snow_data_1), axis=2)

snow_img_num = snow_data_rerange.shape[2]

#------------------ for haze ------------------#
for nn in range(ceil(24 / 8)):
    start = nn * 8
    end = min((nn + 1) * 8, 40)
    if start == 0:
        rainhaze_data_0 = rainhaze_data[start:end, :, :, :]
        gt_rainhaze_data_0 = gt_rainhaze_data[start:end, :, :, :]
    elif start == 8:
        rainhaze_data_1 = rainhaze_data[start:end, :, :, :]
        rainhaze_data_rerange = np.concatenate((rainhaze_data_0, rainhaze_data_1), axis=2)
        gt_rainhaze_data_1 = gt_rainhaze_data[start:end, :, :, :]
        gt_rainhaze_data_rerange = np.concatenate((gt_rainhaze_data_0, gt_rainhaze_data_1), axis=2)
    else:
        rainhaze_data_1 = rainhaze_data[start:end, :, :, :]
        rainhaze_data_rerange = np.concatenate((rainhaze_data_rerange, rainhaze_data_1), axis=2)
        gt_rainhaze_data_1 = gt_rainhaze_data[start:end, :, :, :]
        gt_rainhaze_data_rerange = np.concatenate((gt_rainhaze_data_rerange, gt_rainhaze_data_1), axis=2)

haze_img_num = rainhaze_data_rerange.shape[2]

rain_data = raindrop_data[:9, :, :, :]
gt_rain_data = gt_raindrop_data[:9, :, :, :]
rain_img_num = raindrop_data.shape[2]

allweather_img_num = rain_img_num + snow_img_num + haze_img_num

print(rain_img_num)
print(snow_img_num)
print(haze_img_num)
print(allweather_img_num)

def merge_dims(arr, axis1, axis2):
    """
    将 numpy 数组的两个维度（不要求相邻）合并为一个维度
    arr   : 输入数组
    axis1 : 第一个维度的索引
    axis2 : 第二个维度的索引
    """
    # 规范化（确保 axis1 < axis2）
    if axis1 > axis2:
        axis1, axis2 = axis2, axis1

    shape = list(arr.shape)

    # 新维度的大小
    merged_dim = shape[axis1] * shape[axis2]

    # 先把两个维度挪到相邻（这里把 axis2 移到 axis1+1）
    axes = list(range(arr.ndim))
    axes.pop(axis2)
    axes.insert(axis1 + 1, axis2)
    arr_perm = np.transpose(arr, axes)

    # 构建新 shape
    new_shape = (
            shape[:axis1] +
            [merged_dim] +
            shape[axis1 + 1:axis2] +
            shape[axis2 + 1:]
    )

    return arr_perm.reshape(new_shape)

rain_data = merge_dims(rain_data, 0, 2)  ## 合并 0 和 2 维度 num_patch * 3 * 128 * 128
gt_rain_data = merge_dims(gt_rain_data, 0, 2)  ## 合并 0 和 2 维度 num_patch * 3 * 128 * 128
snow_data = merge_dims(snow_data_rerange, 0, 2)  ## 合并 0 和 2 维度 num_patch * 3 * 128 * 128
gt_snow_data = merge_dims(gt_snow_data_rerange, 0, 2)  ## 合并 0 和 2 维度 num_patch * 3 * 128 * 128
haze_data = merge_dims(rainhaze_data_rerange, 0, 2)  ## 合并 0 和 2 维度 num_patch * 3 * 128 * 128
gt_haze_data = merge_dims(gt_rainhaze_data_rerange, 0, 2)  ## 合并 0 和 2 维度 num_patch * 3 * 128 * 128

allweather_data = np.concatenate((rain_data, snow_data, haze_data), axis=0)
allweather_gt = np.concatenate((gt_rain_data, gt_snow_data, gt_haze_data), axis=0)

assert allweather_data.shape == allweather_gt.shape

new_order = np.random.permutation(allweather_data.shape[0])
allweather_data_rerange = allweather_data[new_order, :, :] ## num_patch * 3 * 128 * 128
allweather_gt_rerange = allweather_gt[new_order, :, :]  ## num_patch * 3 * 128 * 128

from weather_frame_tracker import (
    # make_frame_labels_equal_blocks,
    make_frame_labels_by_blocks,
    apply_same_shuffle_1d,
    FrameCategoryTracker
)

block_lengths = [rain_data.shape[0], snow_data.shape[0], haze_data.shape[0]]  # 按照你原始拼接顺序给出每一类的帧数
allweather_data_labels = make_frame_labels_by_blocks(allweather_data, block_lengths) # 1) 构造“未打乱”的帧标签（不等长块：3 类）
# allweather_gt_labels = make_frame_labels_by_blocks(allweather_gt, block_lengths) # 1) 构造“未打乱”的帧标签（不等长块：3 类）

# 2) 与数据使用相同的 new_order 打乱标签
allweather_data_labels_shuffled = apply_same_shuffle_1d(allweather_data_labels, new_order)  # shape: [T]
allweather_data_labels_shuffled = allweather_data_labels_shuffled.detach().numpy()

allweather_patch_number = rain_data.shape[0] + snow_data.shape[0] + haze_data.shape[0]

for ss in range(ceil(allweather_patch_number / args.batch_size)):
    start = ss * args.batch_size
    end = min((ss + 1) * args.batch_size, allweather_patch_number)
    allweather_data_batch = allweather_data_rerange[start:end, ]  ## batch_size * 3 * 128 * 128

    save_patch = Path(args.train_path) / ('t' + '_rain_' + str(ss + 1) + '.mat')
    savemat(str(save_patch), {'rain_data': allweather_data_batch})

    allweather_data_label_batch = allweather_data_labels_shuffled[start:end]
    save_patch_label = Path(args.train_path) / ('t' + '_label_' + str(ss + 1) + '.mat')
    savemat(str(save_patch_label), {'label_data': allweather_data_label_batch})

for kk in range(ceil(allweather_patch_number / args.batch_size)):
    start = kk * args.batch_size
    end = min((kk + 1) * args.batch_size, allweather_patch_number)
    allweather_gt_batch = allweather_gt_rerange[start:end, ]  ## batch_size * 3 * 128 * 128

    save_patch = Path(args.train_path) / ('t' + '_gt_' + str(kk + 1) + '.mat')
    savemat(str(save_patch), {'gt_data': allweather_gt_batch})

















# def process_images(rain_im_list, rain_gt_floders, args, step_size):
#     """改进后的雨滴图像处理函数"""
#
#     # 预计算所有图像的patch信息
#     patch_info_list = []
#     max_patches = 0
#
#     # 第一遍：收集所有图像的信息
#     for jj, im_path in enumerate(rain_im_list):
#         print('rain:', jj)
#         im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
#         gt_path = rain_gt_floders / (im_path.stem.split('/')[0] + os.path.splitext(im_path)[-1])
#         gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
#
#         c, h, w = im.shape
#         inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size]
#         inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size]
#         num_patches = len(inds_h) * len(inds_w)
#
#         patch_info_list.append({
#             'im': im,
#             'gt': gt,
#             'inds_h': inds_h,
#             'inds_w': inds_w,
#             'num_patches': num_patches
#         })
#
#         max_patches = max(max_patches, num_patches)
#
#     # 使用第一个图像确定统一的patch网格
#     first_info = patch_info_list[0]
#     unified_inds_h = first_info['inds_h']
#     unified_inds_w = first_info['inds_w']
#     unified_num_patches = first_info['num_patches']
#
#     # 预分配内存
#     c = first_info['im'].shape[0]
#     num_frames = len(rain_im_list)
#
#     raindrop_data = np.zeros(shape=[unified_num_patches, c, num_frames,
#                                     args.patch_size, args.patch_size], dtype=np.uint8)
#     gt_raindrop_data = np.zeros(shape=[unified_num_patches, c, num_frames,
#                                        args.patch_size, args.patch_size], dtype=np.uint8)
#
#     # 第二遍：填充数据
#     for frame_idx, patch_info in enumerate(patch_info_list):
#         im = patch_info['im']
#         gt = patch_info['gt']
#
#         # 确保使用统一的patch网格
#         current_inds_h = patch_info['inds_h'][:len(unified_inds_h)]
#         current_inds_w = patch_info['inds_w'][:len(unified_inds_w)]
#
#         iter_patch = 0
#         for i, hh in enumerate(current_inds_h):
#             for j, ww in enumerate(current_inds_w):
#                 if i < len(unified_inds_h) and j < len(unified_inds_w):
#                     raindrop_data[iter_patch, :, frame_idx, :, :] = im[:, hh:hh + args.patch_size, ww:ww + args.patch_size]
#                     gt_raindrop_data[iter_patch, :, frame_idx, :, :] = gt[:, hh:hh + args.patch_size, ww:ww + args.patch_size]
#                     iter_patch += 1
#
#     return raindrop_data, gt_raindrop_data  # number_patch * C * number_image * h * w
#
#
#
# # --------------------for haze----------------------------#
# # 使用改进后的函数
# haze_data, gt_haze_data = process_images(
#     haze_im_list, haze_gt_floders, args, step_size
# )
#
# assert gt_haze_data.shape == haze_data.shape
#
#
# for nn in range(ceil(24 / 8)):
#     start = nn * 8
#     end = min((nn + 1) * 8, 40)
#     if start == 0:
#         haze_data_0 = haze_data[start:end, :, :, :]
#         gt_haze_data_0 = gt_haze_data[start:end, :, :, :]
#     elif start == 8:
#         haze_data_1 = haze_data[start:end, :, :, :]
#         haze_data_rerange = np.concatenate((haze_data_0, haze_data_1), axis=2)
#         gt_haze_data_1 = gt_haze_data[start:end, :, :, :]
#         gt_haze_data_rerange = np.concatenate((gt_haze_data_0, gt_haze_data_1), axis=2)
#     else:
#         haze_data_1 = haze_data[start:end, :, :, :]
#         haze_data_rerange = np.concatenate((haze_data_rerange, haze_data_1), axis=2)
#         gt_haze_data_1 = gt_haze_data[start:end, :, :, :]
#         gt_haze_data_rerange = np.concatenate((gt_haze_data_rerange, gt_haze_data_1), axis=2)
#
#
# haze_img_num = haze_data_rerange.shape[2]
#
#
# # --------------------for snow----------------------------#
# # 使用改进后的函数
# snow_data, gt_snow_data = process_images(
#     snow_im_list, snow_gt_floders, args, step_size
# )
#
# assert gt_snow_data.shape == snow_data.shape
#
#
# for nn in range(ceil(35 / 8)):
#     start = nn * 8
#     end = min((nn + 1) * 8, 35)
#     if start == 0:
#         snow_data_0 = snow_data[start:end, :, :, :]
#         gt_snow_data_0 = gt_snow_data[start:end, :, :, :]
#     elif start == 8:
#         snow_data_1 = snow_data[start:end, :, :, :]
#         snow_data_rerange = np.concatenate((snow_data_0, snow_data_1), axis=2)
#         gt_snow_data_1 = gt_snow_data[start:end, :, :, :]
#         gt_snow_data_rerange = np.concatenate((gt_snow_data_0, gt_snow_data_1), axis=2)
#     else:
#         snow_data_1 = snow_data[start:end, :, :, :]
#         snow_data_rerange = np.concatenate((snow_data_rerange, snow_data_1), axis=2)
#         gt_snow_data_1 = gt_snow_data[start:end, :, :, :]
#         gt_snow_data_rerange = np.concatenate((gt_snow_data_rerange, gt_snow_data_1), axis=2)
#
#
# snow_img_num = snow_data_rerange.shape[2]
#
#
# # --------------------for rain----------------------------#
# # 使用改进后的函数
# rain_data, gt_rain_data = process_images(
#     rain_im_list, rain_gt_floders, args, step_size
# )
#
# assert gt_rain_data.shape == rain_data.shape
#
# rain_data = rain_data[:9, :, :, :]
# gt_rain_data = gt_rain_data[:9, :, :, :]
# rain_img_num = rain_data.shape[2]
#
# allweather_img_num = rain_img_num + snow_img_num + haze_img_num
#
# print(rain_img_num)
# print(snow_img_num)
# print(haze_img_num)
# print(allweather_img_num)
#
#
# def merge_dims(arr, axis1, axis2):
#     """
#     将 numpy 数组的两个维度（不要求相邻）合并为一个维度
#     arr   : 输入数组
#     axis1 : 第一个维度的索引
#     axis2 : 第二个维度的索引
#     """
#     # 规范化（确保 axis1 < axis2）
#     if axis1 > axis2:
#         axis1, axis2 = axis2, axis1
#
#     shape = list(arr.shape)
#
#     # 新维度的大小
#     merged_dim = shape[axis1] * shape[axis2]
#
#     # 先把两个维度挪到相邻（这里把 axis2 移到 axis1+1）
#     axes = list(range(arr.ndim))
#     axes.pop(axis2)
#     axes.insert(axis1 + 1, axis2)
#     arr_perm = np.transpose(arr, axes)
#
#     # 构建新 shape
#     new_shape = (
#             shape[:axis1] +
#             [merged_dim] +
#             shape[axis1 + 1:axis2] +
#             shape[axis2 + 1:]
#     )
#
#     return arr_perm.reshape(new_shape)






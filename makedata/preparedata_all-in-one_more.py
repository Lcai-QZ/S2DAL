import cv2
import argparse
import numpy as np
from pathlib import Path
from scipy.io import savemat
from math import ceil
import random

parser = argparse.ArgumentParser()
parser.add_argument('--allweather_path', type=str, default='/media/zyserver/data16t/cailei/data/weather/allweather',
                    help="Path of the original weather datasets, (default: None)")  ##
parser.add_argument('--train_path', type=str, default='/media/zyserver/data16t/cailei/data/weather/allweather/allweather_train_mat_128_more',
                    help="Path to save the prepared training datasets, (default: None)") ##
parser.add_argument('--patch_size', type=int, default=128,
                    help="Patch Size, (default: None)")  ##
parser.add_argument('--batch_size', type=int, default=6,
                    help="Batch Size, (default: None)")  ## setting batch_size to 3 or 6

args = parser.parse_args()

overlap = int(args.patch_size/4)
step_size = args.patch_size - overlap

# weather data
gt_floder = Path(args.allweather_path) / 'gt'  ##
raindrop_floders = Path(args.allweather_path) / 'raindrop'  ##
raindrop_im_list = sorted([x for x in raindrop_floders.glob('*.png')])   ##
rainhaze_floders = Path(args.allweather_path) / 'rainhaze'  ##
rainhaze_im_list = sorted([x for x in rainhaze_floders.glob('*.png')])   ##
snow_floders = Path(args.allweather_path) / 'snow'  #
snow_im_list = sorted([x for x in snow_floders.glob('*.jpg')])   ##

#--------------------for raindrop----------------------------#
for jj, im_path in enumerate(raindrop_im_list):
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
    gt_path = gt_floder / (im_path.stem.split('.')[0] + '.png')
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
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    gt_path = gt_floder / (im_path.stem.split('.')[0] + '.jpg')
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

#--------------------for rainhaze----------------------------#
for jj, im_path in enumerate(rainhaze_im_list):
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    gt_path = gt_floder / (im_path.stem.split('.')[0] + '.png')
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

# allweather_img_num = raindrop_img_num + snow_img_num + rainhaze_img_num

# print(raindrop_img_num)
# print(snow_img_num)
# print(rainhaze_img_num)
# print(allweather_img_num)

#------------------ for raindrop ------------------#
for nn in range(ceil(40 / 8)):
    start = nn * 8
    end = min((nn + 1) * 8, 40)
    if start == 0:
        raindrop_data_0 = raindrop_data[start:end, :, :, :]
        gt_raindrop_data_0 = gt_raindrop_data[start:end, :, :, :]
    elif start == 8:
        raindrop_data_1 = raindrop_data[start:end, :, :, :]
        raindrop_data_rerange = np.concatenate((raindrop_data_0, raindrop_data_1), axis=2)
        gt_raindrop_data_1 = gt_raindrop_data[start:end, :, :, :]
        gt_raindrop_data_rerange = np.concatenate((gt_raindrop_data_0, gt_raindrop_data_1), axis=2)
    else:
        raindrop_data_1 = raindrop_data[start:end, :, :, :]
        raindrop_data_rerange = np.concatenate((raindrop_data_rerange, raindrop_data_1), axis=2)
        gt_raindrop_data_1 = gt_raindrop_data[start:end, :, :, :]
        gt_raindrop_data_rerange = np.concatenate((gt_raindrop_data_rerange, gt_raindrop_data_1), axis=2)

raindrop_img_num = raindrop_data_rerange.shape[2]

#------------------ for rainhaze ------------------#
for nn in range(ceil(40 / 8)):
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

rainhaze_img_num = rainhaze_data_rerange.shape[2]

snow_data = snow_data[:8, :, :, :]
gt_snow_data = gt_snow_data[:8, :, :, :]
snow_img_num = snow_data.shape[2]

allweather_img_num = raindrop_img_num + snow_img_num + rainhaze_img_num

print(raindrop_img_num)
print(snow_img_num)
print(rainhaze_img_num)
print(allweather_img_num)

for ss in range(ceil(8 / 2)):
    start = ss * 2
    end = min((ss + 1) * 2, 8)
    raindrop_data_batch = raindrop_data_rerange[start:end, ]
    gt_raindrop_data_batch = gt_raindrop_data_rerange[start:end, ]
    snow_data_batch = snow_data[start:end, ]
    gt_snow_data_batch = gt_snow_data[start:end, ]
    rainhaze_data_batch = rainhaze_data_rerange[start:end, ]
    gt_rainhaze_data_batch = gt_rainhaze_data_rerange[start:end, ]
    for nn in range(snow_img_num):
        save_patch = Path(args.train_path) / ('t' + str(nn) + '_rain_' + str(ss + 1) + '.mat')
        save_patch_gt = Path(args.train_path) / ('t' + str(nn) + '_gt_' + str(ss + 1) + '.mat')
        if nn == 0:
            single_raindrop_data_batch = raindrop_data_batch[:, :, :nn + 1, :]
            single_snow_data_batch = snow_data_batch[:, :, :nn + 1, :]
            single_rainhaze_data_batch = rainhaze_data_batch[:, :, :nn + 1, :]
            allweather_data_batch = np.concatenate((single_raindrop_data_batch, single_snow_data_batch, single_rainhaze_data_batch), axis=0)
            savemat(str(save_patch), {'rain_data': allweather_data_batch.squeeze(2)})

            single_gt_raindrop_data_batch = gt_raindrop_data_batch[:, :, :nn + 1, :]
            single_gt_snow_data_batch = gt_snow_data_batch[:, :, :nn + 1, :]
            single_gt_rainhaze_data_batch = gt_rainhaze_data_batch[:, :, :nn + 1, :]
            allweather_gt_data_batch = np.concatenate((single_gt_raindrop_data_batch, single_gt_snow_data_batch, single_gt_rainhaze_data_batch), axis=0)
            savemat(str(save_patch_gt), {'gt_data': allweather_gt_data_batch.squeeze(2)})
        elif nn < raindrop_img_num:
            single_raindrop_data_batch = raindrop_data_batch[:, :, nn:nn + 1, :]
            single_snow_data_batch = snow_data_batch[:, :, nn:nn + 1, :]
            single_rainhaze_data_batch = rainhaze_data_batch[:, :, nn:nn + 1, :]
            allweather_data_batch = np.concatenate((single_raindrop_data_batch, single_snow_data_batch, single_rainhaze_data_batch), axis=0)
            savemat(str(save_patch), {'rain_data': allweather_data_batch.squeeze(2)})

            single_gt_raindrop_data_batch = gt_raindrop_data_batch[:, :, nn:nn + 1, :]
            single_gt_snow_data_batch = gt_snow_data_batch[:, :, nn:nn + 1, :]
            single_gt_rainhaze_data_batch = gt_rainhaze_data_batch[:, :, nn:nn + 1, :]
            allweather_gt_data_batch = np.concatenate((single_gt_raindrop_data_batch, single_gt_snow_data_batch, single_gt_rainhaze_data_batch), axis=0)
            savemat(str(save_patch_gt), {'gt_data': allweather_gt_data_batch.squeeze(2)})
        elif raindrop_img_num <= nn and nn < snow_img_num:
            num_raindrop = random.randint(0, raindrop_img_num - 1)
            single_raindrop_data_batch = raindrop_data_batch[:, :, num_raindrop:num_raindrop + 1, :]
            single_snow_data_batch = snow_data_batch[:, :, nn:nn + 1, :]
            single_rainhaze_data_batch = rainhaze_data_batch[:, :, nn:nn + 1, :]
            allweather_data_batch = np.concatenate((single_raindrop_data_batch, single_snow_data_batch, single_rainhaze_data_batch), axis=0)
            savemat(str(save_patch), {'rain_data': allweather_data_batch.squeeze(2)})

            single_gt_raindrop_data_batch = gt_raindrop_data_batch[:, :, num_raindrop:num_raindrop + 1, :]
            single_gt_snow_data_batch = gt_snow_data_batch[:, :, nn:nn + 1, :]
            single_gt_rainhaze_data_batch = gt_rainhaze_data_batch[:, :, nn:nn + 1, :]
            allweather_gt_data_batch = np.concatenate((single_gt_raindrop_data_batch, single_gt_snow_data_batch, single_gt_rainhaze_data_batch), axis=0)
            savemat(str(save_patch_gt), {'gt_data': allweather_gt_data_batch.squeeze(2)})
        else:
            num_raindrop = random.randint(0, raindrop_img_num - 1)
            single_raindrop_data_batch = raindrop_data_batch[:, :, num_raindrop:num_raindrop + 1, :]
            num_snow = random.randint(0, snow_img_num - 1)
            single_snow_data_batch = snow_data_batch[:, :, num_snow:num_snow + 1, :]
            single_rainhaze_data_batch = rainhaze_data_batch[:, :, nn:nn + 1, :]
            allweather_data_batch = np.concatenate((single_raindrop_data_batch, single_snow_data_batch, single_rainhaze_data_batch), axis=0)
            savemat(str(save_patch), {'rain_data': allweather_data_batch.squeeze(2)})

            single_gt_raindrop_data_batch = gt_raindrop_data_batch[:, :, num_raindrop:num_raindrop + 1, :]
            single_gt_snow_data_batch = gt_snow_data_batch[:, :, num_snow:num_snow + 1, :]
            single_gt_rainhaze_data_batch = gt_rainhaze_data_batch[:, :, nn:nn + 1, :]
            allweather_gt_data_batch = np.concatenate((single_gt_raindrop_data_batch, single_gt_snow_data_batch, single_gt_rainhaze_data_batch), axis=0)
            savemat(str(save_patch_gt), {'gt_data': allweather_gt_data_batch.squeeze(2)})













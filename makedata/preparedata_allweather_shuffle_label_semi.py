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
parser.add_argument('--train_path', type=str, default='/media/zyserver/data16t/cailei/data/weather/allweather/allweather_train_semi_mat_128_label_shuffle',
                    help="Path to save the prepared training datasets, (default: None)") ##
parser.add_argument('--patch_size', type=int, default=128,
                    help="Patch Size, (default: None)")  ##
parser.add_argument('--batch_size', type=int, default=10,
                    help="Batch Size, (default: None)")  ## setting batch_size to 3 or 6

args = parser.parse_args()

overlap = int(args.patch_size/4)
step_size = args.patch_size - overlap

# weather data
raindrop_floders = Path(args.allweather_path) / 'raindrop_semi'  ##
raindrop_im_list = sorted([x for x in raindrop_floders.glob('*.png')])   ##
rainhaze_floders = Path(args.allweather_path) / 'rainhaze_semi'  ##
rainhaze_im_list = sorted([x for x in rainhaze_floders.glob('*.png')])   ##
snow_floders = Path(args.allweather_path) / 'snow_semi'  #
snow_im_list = sorted([x for x in snow_floders.glob('*.jpg')])   ##

#--------------------for raindrop----------------------------#
for jj, im_path in enumerate(raindrop_im_list):
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2,0,1])
    if jj == 0:
        iter_patch = 0
        raindrop_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        c, num_frame, h, w = raindrop_data_temp.shape
        inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        num_patch = len(inds_h) * len(inds_w)
        raindrop_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        # iter_patch = 0
        for hh in inds_h:
            for ww in inds_w:
                raindrop_data[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                iter_patch += 1
        num_frame +=1
    else:
        iter_patch = 0
        raindrop_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        c, _, h, w = raindrop_data_temp.shape
        inds_h_1 = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w_1 = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        # num_patch_1 = len(inds_h_1) * len(inds_w_1)
        if inds_h_1 <= inds_h and inds_w_1 <= inds_w:
            num_patch = len(inds_h_1) * len(inds_w_1)
            inds_h = inds_h_1
            inds_w = inds_w_1
            raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    raindrop_data_1[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            raindrop_data = np.concatenate((raindrop_data, raindrop_data_1), axis=2)
        elif inds_h_1 <= inds_h and inds_w_1 >= inds_w:
            inds_h = inds_h_1
            num_patch = len(inds_h) * len(inds_w)
            raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    raindrop_data_1[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            raindrop_data = np.concatenate((raindrop_data[:iter_patch, :, :, :], raindrop_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 >= inds_h and inds_w_1 <= inds_w:
            inds_w = inds_w_1
            num_patch = len(inds_h) * len(inds_w)
            raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                   raindrop_data_1[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                   iter_patch += 1
            raindrop_data = np.concatenate((raindrop_data[:iter_patch, :, :, :], raindrop_data_1), axis=2)
        else:
            raindrop_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    raindrop_data_1[iter_patch,] = raindrop_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            raindrop_data = np.concatenate((raindrop_data, raindrop_data_1), axis=2)  # c x num_frame x h x w

# assert gt_raindrop_data.shape == raindrop_data.shape
raindrop_img_num = len(raindrop_im_list)

#--------------------for snow----------------------------#
for jj, im_path in enumerate(snow_im_list):
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    if jj == 0:
        iter_patch = 0
        snow_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        c, num_frame, h, w = snow_data_temp.shape
        inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        num_patch = len(inds_h) * len(inds_w)
        snow_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        # iter_patch = 0
        for hh in inds_h:
            for ww in inds_w:
                snow_data[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                iter_patch += 1
        num_frame += 1
    else:
        iter_patch = 0
        snow_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        c, _, h, w = snow_data_temp.shape
        inds_h_1 = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w_1 = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        # num_patch_1 = len(inds_h_1) * len(inds_w_1)
        if inds_h_1 <= inds_h and inds_w_1 <= inds_w:
            num_patch = len(inds_h_1) * len(inds_w_1)
            inds_h = inds_h_1
            inds_w = inds_w_1
            snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    snow_data_1[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            snow_data = np.concatenate((snow_data, snow_data_1), axis=2)
        elif inds_h_1 <= inds_h and inds_w_1 >= inds_w:
            inds_h = inds_h_1
            num_patch = len(inds_h) * len(inds_w)
            snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    snow_data_1[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            snow_data = np.concatenate((snow_data[:iter_patch, :, :, :], snow_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 >= inds_h and inds_w_1 <= inds_w:
            inds_w = inds_w_1
            num_patch = len(inds_h) * len(inds_w)
            snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    snow_data_1[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            snow_data = np.concatenate((snow_data[:iter_patch, :, :, :], snow_data_1), axis=2)
        else:
            snow_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    snow_data_1[iter_patch,] = snow_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            snow_data = np.concatenate((snow_data, snow_data_1), axis=2)  # c x num_frame x h x w

# assert gt_snow_data.shape == snow_data.shape
snow_img_num = len(snow_im_list)

#--------------------for rainhaze----------------------------#
for jj, im_path in enumerate(rainhaze_im_list):
    im = cv2.imread(str(im_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
    if jj == 0:
        iter_patch = 0
        rainhaze_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        c, num_frame, h, w = rainhaze_data_temp.shape
        inds_h = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        num_patch = len(inds_h) * len(inds_w)
        rainhaze_data = np.zeros(shape=[num_patch, c, num_frame, args.patch_size, args.patch_size], dtype=np.uint8)
        # iter_patch = 0
        for hh in inds_h:
            for ww in inds_w:
                rainhaze_data[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                iter_patch += 1
        num_frame +=1
    else:
        iter_patch = 0
        rainhaze_data_temp = im[:, np.newaxis, ]  # c x 1 x h x w
        c, _, h, w = rainhaze_data_temp.shape
        inds_h_1 = list(range(0, h - args.patch_size, step_size)) + [h - args.patch_size, ]
        inds_w_1 = list(range(0, w - args.patch_size, step_size)) + [w - args.patch_size, ]
        # num_patch_1 = len(inds_h_1) * len(inds_w_1)
        if inds_h_1 <= inds_h and inds_w_1 <= inds_w:
            num_patch = len(inds_h_1) * len(inds_w_1)
            inds_h = inds_h_1
            inds_w = inds_w_1
            rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rainhaze_data_1[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rainhaze_data = np.concatenate((rainhaze_data, rainhaze_data_1), axis=2)
        elif inds_h_1 <= inds_h and inds_w_1 >= inds_w:
            inds_h = inds_h_1
            num_patch = len(inds_h) * len(inds_w)
            rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rainhaze_data_1[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rainhaze_data = np.concatenate((rainhaze_data[:iter_patch, :, :, :], rainhaze_data_1), axis=2)  # c x num_frame x h x w
        elif inds_h_1 >= inds_h and inds_w_1 <= inds_w:
            inds_w = inds_w_1
            num_patch = len(inds_h) * len(inds_w)
            rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rainhaze_data_1[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rainhaze_data = np.concatenate((rainhaze_data[:iter_patch, :, :, :], rainhaze_data_1), axis=2)
        else:
            rainhaze_data_1 = np.zeros(shape=[num_patch, c, 1, args.patch_size, args.patch_size], dtype=np.uint8)
            for hh in inds_h:
                for ww in inds_w:
                    rainhaze_data_1[iter_patch,] = rainhaze_data_temp[:, :, hh:hh + args.patch_size, ww:ww + args.patch_size]
                    iter_patch += 1
            rainhaze_data = np.concatenate((rainhaze_data, rainhaze_data_1), axis=2)  # c x num_frame x h x w

# assert gt_rainhaze_data.shape == rainhaze_data.shape
rainhaze_img_num = len(rainhaze_im_list)

allweather_img_num = raindrop_img_num + snow_img_num + rainhaze_img_num

print(raindrop_img_num)
print(snow_img_num)
print(rainhaze_img_num)
print(allweather_img_num)

raindrop_data = raindrop_data[:9, :, :, :] # 9X3Xnum_frameX128X128
snow_data = snow_data[:9, :, :, :]
rainhaze_data = rainhaze_data[:9, :, :, :]

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

raindrop_data = merge_dims(raindrop_data, 0, 2)  ## 合并 0 和 2 维度 num_patch * 3 * 128 * 128
snow_data = merge_dims(snow_data, 0, 2)  ## 合并 0 和 2 维度 num_patch * 3 * 128 * 128
rainhaze_data = merge_dims(rainhaze_data, 0, 2)  ## 合并 0 和 2 维度 num_patch * 3 * 128 * 128

allweather_data = np.concatenate((raindrop_data, snow_data, rainhaze_data), axis=0)

new_order = np.random.permutation(allweather_data.shape[0])
allweather_data_rerange = allweather_data[new_order, :, :] ## num_patch * 3 * 128 * 128

from weather_frame_tracker import (
    # make_frame_labels_equal_blocks,
    make_frame_labels_by_blocks,
    apply_same_shuffle_1d,
    FrameCategoryTracker
)

block_lengths = [raindrop_data.shape[0], snow_data.shape[0], rainhaze_data.shape[0]]  # 按照你原始拼接顺序给出每一类的帧数
allweather_data_labels = make_frame_labels_by_blocks(allweather_data, block_lengths) # 1) 构造“未打乱”的帧标签（不等长块：3 类）

# 2) 与数据使用相同的 new_order 打乱标签
allweather_data_labels_shuffled = apply_same_shuffle_1d(allweather_data_labels, new_order)  # shape: [T]
allweather_data_labels_shuffled = allweather_data_labels_shuffled.detach().numpy()

allweather_patch_number = raindrop_data.shape[0] + snow_data.shape[0] + rainhaze_data.shape[0]

for ss in range(ceil(allweather_patch_number / args.batch_size)):
    start = ss * args.batch_size
    end = min((ss + 1) * args.batch_size, allweather_patch_number)
    allweather_data_batch = allweather_data_rerange[start:end, ]

    save_patch = Path(args.train_path) / ('real' + '_rain_' + str(ss + 1) + '.mat')
    savemat(str(save_patch), {'rain_data': allweather_data_batch})

    allweather_data_label_batch = allweather_data_labels_shuffled[start:end]
    save_patch_label = Path(args.train_path) / ('real' + '_label_' + str(ss + 1) + '.mat')
    savemat(str(save_patch_label), {'label_data': allweather_data_label_batch})



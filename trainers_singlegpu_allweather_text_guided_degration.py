#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-08-29 16:25:28

import os
import sys
import cv2
import shutil
import random
import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
# from networks.derain_net import DerainNet
# from networks.RBNet import RBNet
# from networks.generators_patch128 import GeneratorState, GeneratorRain
from networks.generators_text_guided_degration import GeneratorState, GeneratorRain

from skimage import img_as_float32, img_as_ubyte
from utils import batch_PSNR, batch_SSIM, calculate_parameters
import math
from math import ceil

# pytorch package
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

import importlib
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from vgg_loss import *
from SSIM import SSIM

from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

import os
import random
import torch.nn.functional as F
from functools import partial
import clip

torch.set_default_dtype(torch.float32)

inputext = ["Rain degradation with raindrops",
            "Snow degradation with normal snowflakes",
            "Rain degradation with rain lines and normal haze"] # detailed description.

#inputext[0] = "Rain degradation with raindrops"
# inputext = ["Raindrops","Snowflakes"
#             ,"Rain streaks"] # Simple description

# clip_model, _ = clip.load("ViT-B/32", device=0)
clip_model, _ = clip.load(
    "ViT-B/32",
    device=1,
    download_root="/home/cailei/anaconda3/envs/SAHistoFormer/lib/python3.8/site-packages/clip"
)
for param in clip_model.parameters():
    param.requires_grad = False

class trainer:
    def __init__(self, args, opt):
        '''
        :param args: options
        '''
        # setting random seed
        self.seed = args['seed']
        self.set_seed()

        # collect training data
        self.resume = args['resume']
        self.latent_size = args['latent_size']
        self.state_size = args['state_size']
        self.text_code_size = args['text_code_size']
        self.motion_size = args['motion_size']
        self.latent_dir_name = args['latent_dir_name']
        self.train_path = args['train_path']
        self.train_path_semi = args['train_path_semi']
        self.tidy_train_data()  # self.train_data_list

        # collect testing data
        # self.test_path = args['test_path']
        self.test_path_raindrop = args['raindrop_test_path']
        self.test_path_rainhaze = args['rainhaze_test_path']
        self.test_path_snow_S = args['Snow100K-S_test_path']
        self.test_path_snow_L = args['Snow100K-L_test_path']
        # self.test_path_semi = args['test_path_semi']
        self.tidy_test_data()  # self.test_data, self.test_gt, self.test_data_semi, c x n x h x w, float, torch

        # network settings
        self.patch_size = args['patch_size']
        self.feature_state = args['feature_state']
        self.feature_rain_G = args['feature_rain_G']
        self.n_resblocks = args['n_resblocks']
        self.feature_derain_D = args['feature_derain_D']

        # training settings
        self.rho = args['rho']
        self.tv_weight = args['tv_weight']
        self.epsilon2 = args['epsilon2']
        self.delta = args['delta']
        self.epochs = args['epochs']
        self.resume = args['resume']
        self.lr_D = args['lr_D']
        self.lr_GState = args['lr_GState']
        self.lr_GRain = args['lr_GRain']
        self.weight_decay_D = args['weight_decay_D']
        self.weight_decay_GRain = args['weight_decay_GRain']
        self.weight_decay_GState = args['weight_decay_GState']
        self.milestones = args['milestones']
        self.factor_lr = args['factor_lr']
        self.max_grad_norm_D = args['max_grad_norm_D']
        self.log_dir = args['log_dir']
        self.model_dir = args['model_dir']
        self.max_iter_EM = args['max_iter_EM']
        self.pretrain = args['pretrain']
        self.truncate = args['truncate']
        self.truncate_test = args['truncate_test']
        self.langevin_steps = args['langevin_steps']
        self.print_freq = args['print_freq']

    def set_seed(self):
        print('*' * 150)
        print('Setting random seed: {:d}...'.format(self.seed))
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def tidy_train_data(self):
        print('*' * 150)
        print('Making training data...')

        self.train_data_list = []
        make_latent = True if self.resume is None else False
        latent_dir = Path(self.train_path).parent / self.latent_dir_name
        # print('latent_dir',latent_dir)
        if make_latent:
            if latent_dir.exists():
                shutil.rmtree(str(latent_dir))
            latent_dir.mkdir()

        # labelled data
        allweather_data_list = sorted([str(x) for x in Path(self.train_path).glob('*_rain_*.mat')])
        for allweather_path in allweather_data_list:
            parts_path = Path(allweather_path).name.split('_')
            gt_path = Path(allweather_path).parent / (parts_path[0]+'_gt_'+parts_path[-1])
            label_path = Path(allweather_path).parent / (parts_path[0] + '_label_' + parts_path[-1])

            latent_path = latent_dir / Path(allweather_path).name.replace('rain', 'latent')
            state_path = latent_dir / Path(allweather_path).name.replace('rain', 'state')
            motion_path = latent_dir / Path(allweather_path).name.replace('rain', 'motion')
            generator_path = latent_dir / (Path(allweather_path).stem.replace('rain', 'generator') + '.pt')

            if make_latent:
                allweather_data = loadmat(allweather_path)['rain_data']
                num_batch = allweather_data.shape[:1][0]
                Z = np.random.randn(num_batch, self.latent_size).astype(np.float32)
                savemat(str(latent_path), {'Z': Z})

                S = np.random.randn(num_batch, self.state_size).astype(np.float32)
                savemat(str(state_path), {'S': S})

                M = np.random.randn(num_batch, self.motion_size).astype(np.float32)
                savemat(str(motion_path), {'M': M})

            # self.train_data_list.append({'rainy': allweather_path,
            #                              'gt': str(gt_path),
            #                              'generator': str(generator_path),
            #                              'state': str(state_path),
            #                              'motion': str(motion_path),
            #                              'latent': str(latent_path)})

            self.train_data_list.append({'rainy': allweather_path,
                                         'gt': str(gt_path),
                                         'label': str(label_path),
                                         'generator': str(generator_path),
                                         'state': str(state_path),
                                         'motion': str(motion_path),
                                         'latent': str(latent_path)})

        # unlabelled data
        if self.train_path_semi:
            allweather_data_semi_list = sorted([str(x) for x in Path(self.train_path_semi).glob('*_rain_*.mat')])
            for allweather_path_semi in allweather_data_semi_list:
                parts_path_semi = Path(allweather_path_semi).name.split('_')
                label_path_semi = Path(allweather_path_semi).parent / (parts_path_semi[0] + '_label_' + parts_path_semi[-1])

                latent_path_semi = latent_dir / Path(allweather_path_semi).name.replace('rain', 'latent')
                state_path_semi = latent_dir / Path(allweather_path_semi).name.replace('rain', 'state')
                motion_path_semi = latent_dir / Path(allweather_path_semi).name.replace('rain', 'motion')
                generator_path_semi = latent_dir / (Path(allweather_path_semi).stem.replace('rain', 'generator') + '.pt')

                if make_latent:
                    allweather_data_semi = loadmat(allweather_path_semi)['rain_data']
                    num_batch = allweather_data_semi.shape[:1][0]
                    Z = np.random.randn(num_batch, self.latent_size).astype(np.float32)
                    savemat(str(latent_path_semi), {'Z': Z})

                    S = np.random.randn(num_batch, self.state_size).astype(np.float32)
                    savemat(str(state_path_semi), {'S': S})

                    M = np.random.randn(num_batch, self.motion_size).astype(np.float32)
                    savemat(str(motion_path_semi), {'M': M})

                # self.train_data_list.append({'rainy': allweather_path_semi,
                #                              'generator': str(generator_path_semi),
                #                              'state': str(state_path_semi),
                #                              'motion': str(motion_path_semi),
                #                              'latent': str(latent_path_semi)})

                self.train_data_list.append({'rainy': allweather_path_semi,
                                             'label': str(label_path_semi),
                                             'generator': str(generator_path_semi),
                                             'state': str(state_path_semi),
                                             'motion': str(motion_path_semi),
                                             'latent': str(latent_path_semi)})

        random.shuffle(self.train_data_list)

    def tidy_test_data(self):
        print('*' * 150)
        print('Making testing data...')

        # -------------------- for test raindrop----------------------#
        test_raindrop_data_list = sorted([x for x in Path(self.test_path_raindrop).glob('*.png')])
        # num = len(test_data_list)
        for ii, raindrop_path in enumerate(test_raindrop_data_list):
            gt_name = raindrop_path.stem.split('_')[0] + '_clean.png'
            gt_path = Path(self.test_path_raindrop).parent / Path(self.test_path_raindrop).stem.replace('input', 'gt') / gt_name  #

            im_raindrop = cv2.imread(str(raindrop_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
            im_gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
            if ii == 0:
                H = im_gt.shape[2]
                W = im_gt.shape[1]
                int_H = random.randint(0, H - 128)
                int_W = random.randint(0, W - 128)
                im_raindrop = im_raindrop[:, int_W:int_W + 128, int_H:int_H + 128]  ## randomly cropped
                im_gt = im_gt[:, int_W:int_W + 128, int_H:int_H + 128]
                test_data = im_raindrop[np.newaxis,]
                test_gt = im_gt[np.newaxis,]
            else:
                H = im_gt.shape[2]
                W = im_gt.shape[1]
                int_H = random.randint(0, H - 128)
                int_W = random.randint(0, W - 128)
                im_raindrop = im_raindrop[:, int_W:int_W + 128, int_H:int_H + 128]  ## randomly cropped
                im_gt = im_gt[:, int_W:int_W + 128, int_H:int_H + 128]
                test_data = np.concatenate((test_data, im_raindrop[np.newaxis,]), axis=0)
                test_gt = np.concatenate((test_gt, im_gt[np.newaxis,]), axis=0)

        self.test_raindrop_gt = torch.from_numpy(img_as_float32(test_gt))
        self.test_raindrop_data = torch.from_numpy(img_as_float32(test_data))

        # -------------------- for test rainhaze----------------------#
        test_rainhaze_data_list = sorted([x for x in Path(self.test_path_rainhaze).glob('*.png')])
        # num = len(test_data_list)
        for ii, rainhaze_path in enumerate(test_rainhaze_data_list):
            gt_name = rainhaze_path.stem.split('.')[0] + '.png'
            gt_path = Path(self.test_path_rainhaze).parent / Path(self.test_path_rainhaze).stem.replace('input',
                                                                                                        'gt') / gt_name  #
            im_rainhaze = cv2.imread(str(rainhaze_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
            im_gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
            if ii == 0:
                H = im_gt.shape[2]
                W = im_gt.shape[1]
                int_H = random.randint(0, H - 128)
                int_W = random.randint(0, W - 128)
                im_rainhaze = im_rainhaze[:, int_W:int_W + 128, int_H:int_H + 128]  ## randomly cropped
                im_gt = im_gt[:, int_W:int_W + 128, int_H:int_H + 128]
                test_data = im_rainhaze[np.newaxis,]
                test_gt = im_gt[np.newaxis,]
            else:
                H = im_gt.shape[2]
                W = im_gt.shape[1]
                int_H = random.randint(0, H - 128)
                int_W = random.randint(0, W - 128)
                im_rainhaze = im_rainhaze[:, int_W:int_W + 128, int_H:int_H + 128]  ## randomly cropped
                im_gt = im_gt[:, int_W:int_W + 128, int_H:int_H + 128]
                test_data = np.concatenate((test_data, im_rainhaze[np.newaxis,]), axis=0)
                test_gt = np.concatenate((test_gt, im_gt[np.newaxis,]), axis=0)

        self.test_rainhaze_gt = torch.from_numpy(img_as_float32(test_gt))
        self.test_rainhaze_data = torch.from_numpy(img_as_float32(test_data))

        # -------------------- for test snow-S----------------------#
        test_snow_S_data_list = sorted([x for x in Path(self.test_path_snow_S).glob('*.jpg')])
        # num = len(test_data_list)
        for ii, snow_S_path in enumerate(test_snow_S_data_list):
            gt_name = snow_S_path.stem.split('.')[0] + '.jpg'
            gt_path = Path(self.test_path_snow_S).parent / Path(self.test_path_snow_S).stem.replace('synthetic',
                                                                                                    'gt') / gt_name  #
            im_snow_S = cv2.imread(str(snow_S_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
            im_gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
            if ii == 0:
                H = im_gt.shape[2]
                W = im_gt.shape[1]
                int_H = random.randint(0, H - 128)
                int_W = random.randint(0, W - 128)
                im_snow_S = im_snow_S[:, int_W:int_W + 128, int_H:int_H + 128]  ## randomly cropped
                im_gt = im_gt[:, int_W:int_W + 128, int_H:int_H + 128]
                test_data = im_snow_S[np.newaxis,]
                test_gt = im_gt[np.newaxis,]
            else:
                H = im_gt.shape[2]
                W = im_gt.shape[1]
                int_H = random.randint(0, H - 128)
                int_W = random.randint(0, W - 128)
                im_snow_S = im_snow_S[:, int_W:int_W + 128, int_H:int_H + 128]  ## randomly cropped
                im_gt = im_gt[:, int_W:int_W + 128, int_H:int_H + 128]
                test_data = np.concatenate((test_data, im_snow_S[np.newaxis,]), axis=0)
                test_gt = np.concatenate((test_gt, im_gt[np.newaxis,]), axis=0)

        self.test_snow_S_gt = torch.from_numpy(img_as_float32(test_gt))
        self.test_snow_S_data = torch.from_numpy(img_as_float32(test_data))

        # -------------------- for test snow-L----------------------#
        test_snow_L_data_list = sorted([x for x in Path(self.test_path_snow_L).glob('*.jpg')])
        # num = len(test_data_list)
        for ii, snow_L_path in enumerate(test_snow_L_data_list):
            gt_name = snow_L_path.stem.split('.')[0] + '.jpg'
            gt_path = Path(self.test_path_snow_L).parent / Path(self.test_path_snow_L).stem.replace('synthetic',
                                                                                                    'gt') / gt_name  #
            im_snow_L = cv2.imread(str(snow_L_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
            im_gt = cv2.imread(str(gt_path), flags=cv2.IMREAD_COLOR)[:, :, ::-1].transpose([2, 0, 1])
            if ii == 0:
                H = im_gt.shape[2]
                W = im_gt.shape[1]
                int_H = random.randint(0, H - 128)
                int_W = random.randint(0, W - 128)
                im_snow_L = im_snow_L[:, int_W:int_W + 128, int_H:int_H + 128]  ## randomly cropped
                im_gt = im_gt[:, int_W:int_W + 128, int_H:int_H + 128]
                test_data = im_snow_L[np.newaxis,]
                test_gt = im_gt[np.newaxis,]
            else:
                H = im_gt.shape[2]
                W = im_gt.shape[1]
                int_H = random.randint(0, H - 128)
                int_W = random.randint(0, W - 128)
                im_snow_L = im_snow_L[:, int_W:int_W + 128, int_H:int_H + 128]  ## randomly cropped
                im_gt = im_gt[:, int_W:int_W + 128, int_H:int_H + 128]
                test_data = np.concatenate((test_data, im_snow_L[np.newaxis,]), axis=0)
                test_gt = np.concatenate((test_gt, im_gt[np.newaxis,]), axis=0)

        self.test_snow_L_gt = torch.from_numpy(img_as_float32(test_gt))
        self.test_snow_L_data = torch.from_numpy(img_as_float32(test_data))


    def build_network(self, opt):
        self.GStateNet = GeneratorState(latent_size=self.latent_size,
                                        motion_size=self.motion_size,
                                        num_feature=self.feature_state).cuda()
        # self.GStateNet = GeneratorState(latent_size=self.latent_size,
        #                                 motion_size=self.motion_size,
        #                                 text_size=self.text_code_size,
        #                                 num_feature=self.feature_state).cuda()
        # self.GRainNet = GeneratorRain(im_size=[self.patch_size, ] * 2,
        #                               out_channels=3,
        #                               state_size=self.state_size,
        #                               num_feature=self.feature_rain_G).cuda()
        self.GRainNet = GeneratorRain(im_size=[self.patch_size, ] * 2,
                                      out_channels=3,
                                      state_size=self.state_size,
                                      text_size=self.text_code_size,
                                      num_feature=self.feature_rain_G).cuda()
        self.RestorationNet = define_network(deepcopy(opt['network_g'])).cuda()
        print('*' * 150)
        print('Number of parameters in Derain Net:{:d}'.format(calculate_parameters(self.RestorationNet)))

    def decay_lr(self, ii):
        for stone in self.milestones:
            if (ii + 1) == stone:
                self.optimizerD.param_groups[0]['lr'] *= self.factor_lr

    def load_checkpoint(self):
        if self.resume is not None:
            print('Loading checkpoint from {:s}'.format(self.resume))
            checkpoint_D = torch.load(self.resume)
            self.RestorationNet.load_state_dict(checkpoint_D['DNet'])

            self.start_epoch = checkpoint_D['epoch']
            self.log_im_step = checkpoint_D['step_img']
            self.log_loss_step = checkpoint_D['step_loss']
            self.max_grad_norm_D = checkpoint_D['max_grad_norm_D']
            for ii in range(self.start_epoch):
                self.decay_lr(ii)
        else:
            self.start_epoch = 0
            self.log_loss_step = 0
            self.log_im_step = {'train': 0, 'test': 0}
            # path to save log
            if Path(self.log_dir).is_dir():
                shutil.rmtree(str(Path(self.log_dir)))
            Path(self.log_dir).mkdir()

            # path to save model
            if Path(self.model_dir).is_dir():
                shutil.rmtree(str(Path(self.model_dir)))
            Path(self.model_dir).mkdir()

    @staticmethod
    def load_data_video(current_path, semi=True):
        Y = loadmat(current_path['rainy'])['rain_data']  # num_batch x c x num_frame x p x p
        L = loadmat(current_path['label'])['label_data']
        Z = loadmat(current_path['latent'])['Z']  # num_batch x num_frame x latent_size
        S = loadmat(current_path['state'])['S']  # num_batch x state_size
        M = loadmat(current_path['motion'])['M']  # num_batch x state_size
        if not semi:
            Y_gt = loadmat(current_path['gt'])['gt_data']  # num_batch x c x num_frame x p x p
            return Y, Y_gt, Z, S, M, L
        else:
            return Y, Z, S, M, L

    @staticmethod
    def tv1_norm2d(x, weight):
        '''
        Tv norm.
        :param x: B x 3 x num_frame x p x p
        :param weight: list with length 3
        '''
        # B, C, N = x.shape[:3]
        B, C = x.shape[:2]
        # x_tv = (x[:, :, :, 1:, :] - x[:, :, :, :-1, :]).abs().sum() * weight[0]
        x_tv = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().sum() * weight[0]
        # y_tv = (x[:, :, :, :, 1:] - x[:, :, :, :, :-1]).abs().sum() * weight[1]
        y_tv = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().sum() * weight[1]
        # z_tv = (x[:, :, 1:, :, :] - x[:, :, :-1, :, :]).abs().sum() * weight[2]
        # tv_loss = (x_tv + y_tv + z_tv) / (B * C * N)
        tv_loss = (x_tv + y_tv) / (B * C)
        return tv_loss

    def G_forward_truncate(self, truncate_Z, text_code, motion_type):
        '''
        Forward propagation of Generator for truncated data.
        :param truncate_Z: Batch x num_frame x latent_size tensor
        :param initial_state:  Batch x state_size tensor
        :param motion_type:  Batch x state_size tensor
        '''
        # rain_gen_all = []
        # state_next = initial_state
        # B, num_frame = truncate_Z.shape[:2]
        B = truncate_Z.shape[:1][0]
        input_Z = truncate_Z[:, :].view([B, -1])
        state_next = self.GStateNet(input_Z, motion_type)  # B x state_size
        # rain_gen = self.GRainNet(state_next)
        rain_gen = self.GRainNet(state_next, text_code)
        # rain_gen_all.append(rain_gen)
        # for ii in range(num_frame):
        #     input_Z = truncate_Z[:, ii, :].view([B,-1])
        #     # state_next = self.GStateNet(input_Z, state_next, motion_type)  # B x state_size
        #     state_next = self.GStateNet(input_Z, motion_type)  # B x state_size
        #     rain_gen = self.GRainNet(state_next)
        #     rain_gen_all.append(rain_gen)

        # return torch.stack(rain_gen_all, dim=2), state_next
        return rain_gen, state_next

    def get_loss_MStep(self, Y, back_pre, rain_gen, gt):
        '''
        :param Y: B x 3 x num_frame x p x p tensor, rainy video
        :param back_pre: B x 3 x num_frame x p x p tensor, derained video
        :param rain_gen: B x 3 x num_frame x p x p tensor, generated rain
        :param gt: B x 3 x num_frame x p x p tensor, groundtruth video
        '''
        sigma = (Y - back_pre.detach() - rain_gen.detach()).flatten().std().item()
        likelihood = 0.5 / (sigma ** 2) * (Y - back_pre - rain_gen).square().mean()
        tv_loss = self.rho * self.tv1_norm2d(back_pre, self.tv_weight)
        if gt is None:
            mse_scale = torch.tensor(0)
        else:
            mse_scale = 0.5 / self.epsilon2 * (back_pre - gt).square().mean()
        loss = likelihood + mse_scale + tv_loss
        return loss, likelihood, mse_scale, tv_loss

    def get_loss_MIStep(self, Y, back_pre, rain_gen, gt):
        '''
        :param Y: B x 3 x num_frame x p x p tensor, rainy video
        :param back_pre: B x 3 x num_frame x p x p tensor, derained video
        :param rain_gen: B x 3 x num_frame x p x p tensor, generated rain
        :param gt: B x 3 x num_frame x p x p tensor, groundtruth video
        '''
        sigma = (Y - back_pre.detach() - rain_gen.detach()).flatten().std().item()
        likelihood = 0.5 / (sigma ** 2) * (Y - back_pre - rain_gen).square().mean()
        tv_loss = self.rho * self.tv1_norm2d(back_pre, self.tv_weight)
        # criterion = SSIM().cuda()
        if gt is None:
            mse_scale = torch.tensor(0)
            ssim_loss = torch.tensor(0)
        else:
            mse_scale = 0.5 / self.epsilon2 * (back_pre - gt).square().mean()
            ssim_loss = 0.05 / self.epsilon2 * (1.0 - self.criterion(back_pre, gt))
        loss = likelihood + mse_scale + tv_loss + ssim_loss
        return loss, likelihood, mse_scale, tv_loss, ssim_loss

    def get_loss_MRStep(self, Y, back_pre, rain_gen, gt):
        '''
        :param Y: B x 3 x num_frame x p x p tensor, rainy video
        :param back_pre: B x 3 x num_frame x p x p tensor, derained video
        :param rain_gen: B x 3 x num_frame x p x p tensor, generated rain
        :param gt: B x 3 x num_frame x p x p tensor, groundtruth video
        '''
        sigma = (Y - back_pre.detach() - rain_gen.detach()).flatten().std().item()
        likelihood = 0.5 / (sigma ** 2) * (Y - back_pre - rain_gen).square().mean()
        tv_loss = self.rho * self.tv1_norm2d(back_pre, self.tv_weight)
        if gt is None:
            mse_scale = torch.tensor(0)
            cor_loss = torch.tensor(0)
            ssim_loss = torch.tensor(0)
        else:
            mse_scale = 0.5 / self.epsilon2 * (back_pre - gt).square().mean()
            ssim_loss = 0.05 / self.epsilon2 * (1.0 - self.criterion(back_pre, gt))
            b, c = back_pre.shape[0:2]
            # x1 = back_pre.view(b, -1)
            x1 = back_pre.reshape(b, -1)
            # x2 = gt.contiguous().view(b, -1)
            x2 = gt.reshape(b, -1)
            pearson = (1. - self.pearson_correlation_loss(x1, x2)) / 2.
            # cor_loss = 5. * pearson[~pearson.isnan() * ~pearson.isinf()].mean()
            cor_loss = 0.5 / self.epsilon2 * pearson[~pearson.isnan() * ~pearson.isinf()].mean()
        loss = likelihood + cor_loss + mse_scale + tv_loss + ssim_loss
        return loss, likelihood, mse_scale, tv_loss, cor_loss, ssim_loss

    def pearson_correlation_loss(self, x1, x2):
        assert x1.shape == x2.shape
        b, c = x1.shape[:2]
        dim = -1
        x1, x2 = x1.reshape(b, -1), x2.reshape(b, -1)
        x1_mean, x2_mean = x1.mean(dim=dim, keepdims=True), x2.mean(dim=dim, keepdims=True)
        numerator = ((x1 - x1_mean) * (x2 - x2_mean)).sum(dim=dim, keepdims=True)

        std1 = (x1 - x1_mean).pow(2).sum(dim=dim, keepdims=True).sqrt()
        std2 = (x2 - x2_mean).pow(2).sum(dim=dim, keepdims=True).sqrt()
        denominator = std1 * std2
        corr = numerator.div(denominator + 1e-6)
        return corr

    @staticmethod
    def get_loss_EStep(rain_gt, rain_gen):
        '''
        :param rain_gt: B x 3 x num_frame x p x p tensor, pesudoe rain layer groundtruth
        :param rain_gen: B x 3 x num_frame x p x p tensor, generated rain
        '''
        B, _, N = rain_gt.shape[:3]
        sigma = (rain_gt - rain_gen.detach()).flatten().std().item()
        loss = 0.5 / (sigma ** 2) * (rain_gt - rain_gen).square().sum()
        loss /= (B * N)
        return loss

    def freeze_Generator(self):
        for param in self.GStateNet.parameters():
            param.requires_grad = False
        for param in self.GRainNet.parameters():
            param.requires_grad = False

    def unfreeze_Generator(self):
        for param in self.GStateNet.parameters():
            param.requires_grad = True
        for param in self.GRainNet.parameters():
            param.requires_grad = True

    def predict_deraining(self, args):
        self.RestorationNet.eval()

        # --------------------------for testing raindrop------------------------------#
        # current_data_list = [self.test_raindrop_data, self.test_rainhaze_data, self.test_snow_S_data, self.test_snow_L_data]
        current_data_list = [self.test_raindrop_data, ]
        for kk, current_data in enumerate(current_data_list):
            num_frame = current_data.shape[0]
            test_data = torch.zeros(current_data.shape)
            for ii in range(ceil(num_frame / self.truncate_test)):
                start_ind = ii * self.truncate_test
                end_ind = min((ii + 1) * self.truncate_test, num_frame)
                inputs = current_data[start_ind:end_ind, ].cuda()

                img_id = np.zeros(inputs.shape[0], dtype=int)
                text_prompt_list = [inputext[idx] for idx in img_id]
                # text_token = clip.tokenize(text_prompt_list).to(1)
                text_token = clip.tokenize(text_prompt_list).cuda()
                # text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)
                text_code = clip_model.encode_text(text_token).to(dtype=torch.float32).cuda()

                # start_ind = ii * self.truncate_test
                # end_ind = min((ii + 1) * self.truncate_test, num_frame)
                # inputs = current_data[start_ind:end_ind, ].cuda()
                with torch.set_grad_enabled(False):
                    out = self.RestorationNet(inputs, text_code).clamp_(0.0, 1.0).squeeze(0)
                test_data[start_ind:end_ind, ] = out.cpu()

                if len(current_data_list) == 2 and kk == 1:
                    # x1 = vutils.make_grid(inputs.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                    x1 = vutils.make_grid(inputs, normalize=True, scale_each=True)
                    self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                    # x2 = vutils.make_grid(out.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                    x2 = vutils.make_grid(out, normalize=True, scale_each=True)
                    self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                    self.log_im_step['test'] += 1
                else:
                    if random.randint(1, 10) == 1:
                        # x1 = vutils.make_grid(inputs.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                        x1 = vutils.make_grid(inputs, normalize=True, scale_each=True)
                        self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                        # x2 = vutils.make_grid(out.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                        x2 = vutils.make_grid(out, normalize=True, scale_each=True)
                        self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                        self.log_im_step['test'] += 1

                if kk == 0:
                    # self.psnrm = batch_PSNR(test_data_derain[:, 2:-2, ].permute([1, 0, 2, 3]), self.test_gt[:, 2:-2, ].permute([1, 0, 2, 3]), ycbcr=False)
                    self.psnrm_raindrop = batch_PSNR(test_data, self.test_raindrop_gt, ycbcr=False)
                    # self.ssimm = batch_SSIM(test_data_derain[:, 2:-2, ].permute([1, 0, 2, 3]), self.test_gt[:, 2:-2, ].permute([1, 0, 2, 3]), ycbcr=False)
                    self.ssimm_raindrop = batch_SSIM(test_data, self.test_raindrop_gt, ycbcr=False)

        # --------------------------for testing rainhaze------------------------------#
        # current_data_list = [self.test_raindrop_data, self.test_rainhaze_data, self.test_snow_S_data, self.test_snow_L_data]
        current_data_list = [self.test_rainhaze_data, ]
        for kk, current_data in enumerate(current_data_list):
            num_frame = current_data.shape[0]
            test_data = torch.zeros(current_data.shape)  # c x n x p x p
            for ii in range(ceil(num_frame / self.truncate_test)):
                start_ind = ii * self.truncate_test
                end_ind = min((ii + 1) * self.truncate_test, num_frame)
                inputs = current_data[start_ind:end_ind, ].cuda()


                # img_id = np.full(self.truncate_test, 2)
                img_id = np.full(inputs.shape[0], 2)
                text_prompt_list = [inputext[idx] for idx in img_id]
                # text_token = clip.tokenize(text_prompt_list).to(1)
                text_token = clip.tokenize(text_prompt_list).cuda()
                # text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)
                text_code = clip_model.encode_text(text_token).to(dtype=torch.float32).cuda()

                with torch.set_grad_enabled(False):
                    out = self.RestorationNet(inputs, text_code).clamp_(0.0, 1.0).squeeze(0)
                test_data[start_ind:end_ind, ] = out.cpu()

                if len(current_data_list) == 2 and kk == 1:
                    # x1 = vutils.make_grid(inputs.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                    x1 = vutils.make_grid(inputs, normalize=True, scale_each=True)
                    self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                    # x2 = vutils.make_grid(out.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                    x2 = vutils.make_grid(out, normalize=True, scale_each=True)
                    self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                    self.log_im_step['test'] += 1
                else:
                    if random.randint(1, 10) == 1:
                        # x1 = vutils.make_grid(inputs.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                        x1 = vutils.make_grid(inputs, normalize=True, scale_each=True)
                        self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                        # x2 = vutils.make_grid(out.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                        x2 = vutils.make_grid(out, normalize=True, scale_each=True)
                        self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                        self.log_im_step['test'] += 1

            if kk == 0:
                # self.psnrm = batch_PSNR(test_data_derain[:, 2:-2, ].permute([1, 0, 2, 3]), self.test_gt[:, 2:-2, ].permute([1, 0, 2, 3]), ycbcr=False)
                self.psnrm_rainhaze = batch_PSNR(test_data, self.test_rainhaze_gt, ycbcr=False)
                # self.ssimm = batch_SSIM(test_data_derain[:, 2:-2, ].permute([1, 0, 2, 3]), self.test_gt[:, 2:-2, ].permute([1, 0, 2, 3]), ycbcr=False)
                self.ssimm_rainhaze = batch_SSIM(test_data, self.test_rainhaze_gt, ycbcr=False)

        # --------------------------for testing Snow100K-S------------------------------#
        # current_data_list = [self.test_raindrop_data, self.test_rainhaze_data, self.test_snow_S_data, self.test_snow_L_data]
        current_data_list = [self.test_snow_S_data, ]
        for kk, current_data in enumerate(current_data_list):
            num_frame = current_data.shape[0]
            test_data = torch.zeros(current_data.shape)  # c x n x p x p
            for ii in range(ceil(num_frame / self.truncate_test)):
                start_ind = ii * self.truncate_test
                end_ind = min((ii + 1) * self.truncate_test, num_frame)
                inputs = current_data[start_ind:end_ind, ].cuda()

                # img_id = np.ones(self.truncate_test, dtype=int)
                img_id = np.ones(inputs.shape[0], dtype=int)
                text_prompt_list = [inputext[idx] for idx in img_id]
                # text_token = clip.tokenize(text_prompt_list).to(1)
                text_token = clip.tokenize(text_prompt_list).cuda()
                # text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)
                text_code = clip_model.encode_text(text_token).to(dtype=torch.float32).cuda()

                with torch.set_grad_enabled(False):
                    out = self.RestorationNet(inputs, text_code).clamp_(0.0, 1.0).squeeze(0)
                test_data[start_ind:end_ind, ] = out.cpu()

                if len(current_data_list) == 2 and kk == 1:
                    # x1 = vutils.make_grid(inputs.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                    x1 = vutils.make_grid(inputs, normalize=True, scale_each=True)
                    self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                    # x2 = vutils.make_grid(out.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                    x2 = vutils.make_grid(out, normalize=True, scale_each=True)
                    self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                    self.log_im_step['test'] += 1
                else:
                    if random.randint(1, 10) == 1:
                        # x1 = vutils.make_grid(inputs.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                        x1 = vutils.make_grid(inputs, normalize=True, scale_each=True)
                        self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                        # x2 = vutils.make_grid(out.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                        x2 = vutils.make_grid(out, normalize=True, scale_each=True)
                        self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                        self.log_im_step['test'] += 1

            if kk == 0:
                # self.psnrm = batch_PSNR(test_data_derain[:, 2:-2, ].permute([1, 0, 2, 3]), self.test_gt[:, 2:-2, ].permute([1, 0, 2, 3]), ycbcr=False)
                self.psnrm_snow_S = batch_PSNR(test_data, self.test_snow_S_gt, ycbcr=False)
                # self.ssimm = batch_SSIM(test_data_derain[:, 2:-2, ].permute([1, 0, 2, 3]), self.test_gt[:, 2:-2, ].permute([1, 0, 2, 3]), ycbcr=False)
                self.ssimm_snow_S = batch_SSIM(test_data, self.test_snow_S_gt, ycbcr=False)

        # --------------------------for testing Snow100K-L------------------------------#
        # current_data_list = [self.test_raindrop_data, self.test_rainhaze_data, self.test_snow_S_data, self.test_snow_L_data]
        current_data_list = [self.test_snow_L_data, ]
        for kk, current_data in enumerate(current_data_list):
            num_frame = current_data.shape[0]
            test_data = torch.zeros(current_data.shape)  # c x n x p x p
            for ii in range(ceil(num_frame / self.truncate_test)):
                start_ind = ii * self.truncate_test
                end_ind = min((ii + 1) * self.truncate_test, num_frame)
                inputs = current_data[start_ind:end_ind, ].cuda()

                # img_id = np.ones(self.truncate_test, dtype=int)
                img_id = np.ones(inputs.shape[0], dtype=int)
                text_prompt_list = [inputext[idx] for idx in img_id]
                # text_token = clip.tokenize(text_prompt_list).to(1)
                text_token = clip.tokenize(text_prompt_list).cuda()
                # text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)
                text_code = clip_model.encode_text(text_token).to(dtype=torch.float32).cuda()

                with torch.set_grad_enabled(False):
                    out = self.RestorationNet(inputs, text_code).clamp_(0.0, 1.0).squeeze(0)
                test_data[start_ind:end_ind, ] = out.cpu()

                if len(current_data_list) == 2 and kk == 1:
                    # x1 = vutils.make_grid(inputs.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                    x1 = vutils.make_grid(inputs, normalize=True, scale_each=True)
                    self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                    # x2 = vutils.make_grid(out.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                    x2 = vutils.make_grid(out, normalize=True, scale_each=True)
                    self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                    self.log_im_step['test'] += 1
                else:
                    if random.randint(1, 10) == 1:
                        # x1 = vutils.make_grid(inputs.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                        x1 = vutils.make_grid(inputs, normalize=True, scale_each=True)
                        self.writer.add_image('Test Rainy Image', x1, self.log_im_step['test'])
                        # x2 = vutils.make_grid(out.permute([1, 0, 2, 3]), normalize=True, scale_each=True)
                        x2 = vutils.make_grid(out, normalize=True, scale_each=True)
                        self.writer.add_image('Test Deained Image', x2, self.log_im_step['test'])
                        self.log_im_step['test'] += 1

            if kk == 0:
                # self.psnrm = batch_PSNR(test_data_derain[:, 2:-2, ].permute([1, 0, 2, 3]), self.test_gt[:, 2:-2, ].permute([1, 0, 2, 3]), ycbcr=False)
                self.psnrm_snow_L = batch_PSNR(test_data, self.test_snow_L_gt, ycbcr=False)
                # self.ssimm = batch_SSIM(test_data_derain[:, 2:-2, ].permute([1, 0, 2, 3]), self.test_gt[:, 2:-2, ].permute([1, 0, 2, 3]), ycbcr=False)
                self.ssimm_snow_L = batch_SSIM(test_data, self.test_snow_L_gt, ycbcr=False)

        self.psnrm_avg = (self.psnrm_snow_L + self.psnrm_snow_S + self.psnrm_rainhaze + self.psnrm_raindrop) / 4.0
        self.ssimm_avg = (self.ssimm_snow_L + self.ssimm_snow_S + self.ssimm_rainhaze + self.ssimm_raindrop) / 4.0

    def train(self, args, opt):
        # setting visible gpu
        if torch.cuda.is_available():
            torch.cuda.set_device(1)

        # build network
        self.build_network(opt)

        # optimizer
        self.optimizerD = optim.Adam(self.RestorationNet.parameters(),
                                     lr=self.lr_D,
                                     weight_decay=self.weight_decay_D,
                                     betas=(0.5, 0.999))

        # Loading from one specific checkpoint
        self.load_checkpoint()
        self.criterion = SSIM().cuda()

        # open the tensorboard
        self.writer = SummaryWriter(str(Path(self.log_dir)))

        best_psnr_raindrop = 0
        best_psnr_raindrop_epoch = 0
        best_ssim_raindrop = 0
        best_ssim_raindrop_epoch = 0

        best_psnr_rainhaze = 0
        best_psnr_rainhaze_epoch = 0
        best_ssim_rainhaze = 0
        best_ssim_rainhaze_epoch = 0

        best_psnr_snow_S = 0
        best_psnr_snow_S_epoch = 0
        best_ssim_snow_S = 0
        best_ssim_snow_S_epoch = 0

        best_psnr_snow_L = 0
        best_psnr_snow_L_epoch = 0
        best_ssim_snow_L = 0
        best_ssim_snow_L_epoch = 0

        best_psnr_avg = 0
        best_psnr_avg_epoch = 0
        best_ssim_avg = 0
        best_ssim_avg_epoch = 0

        # begin training
        for ii in range(self.start_epoch, self.epochs):
            self.RestorationNet.train()

            lossM_epoch = likelihood_epoch = mse_epoch = tv_epoch = criter_epoch = ssim_epoch = 0
            mean_norm_grad_epoch_D = 0

            for jj, current_path in enumerate(self.train_data_list):
                if ii >= self.pretrain:
                    checkpoint_path_G = current_path['generator']
                    checkpoint_G = torch.load(checkpoint_path_G)
                    self.GStateNet.load_state_dict(checkpoint_G['GState'])
                    self.GRainNet.load_state_dict(checkpoint_G['GRain'])

                optimizerG = optim.Adam([{'params': self.GStateNet.parameters(),
                                          'lr': self.lr_GState,
                                          'weight_decay': self.weight_decay_GState},
                                         {'params': self.GRainNet.parameters(),
                                          'lr': self.lr_GRain,
                                          'weight_decay': self.weight_decay_GRain}],
                                        betas=(0.5, 0.999))

                # load data
                if 'gt' in current_path:
                    Y, Y_gt, Z, S, M, L = self.load_data_video(current_path, semi=False)
                else:
                    Y, Z, S, M, L = self.load_data_video(current_path, semi=True)
                assert self.patch_size == Y.shape[-1]

                num_batch = Y.shape[:1][0]
                lossM_batch = likelihood_batch = mse_batch = tv_batch = criter_batch = ssim_batch = 0
                mean_norm_grad_batch_D = 0
                input_M = torch.from_numpy(M).cuda()

                inputs = torch.from_numpy(img_as_float32(Y)).cuda()
                # img_id = np.array([0, 1, 2])
                # img_id = np.array([0, 0, 1, 1, 2, 2])
                img_id = L.flatten()
                text_prompt_list = [inputext[idx] for idx in img_id]
                # text_token = clip.tokenize(text_prompt_list).to(1)
                text_token = clip.tokenize(text_prompt_list).cuda()
                # text_code = clip_model.encode_text(text_token).to(dtype=torch.float32)
                text_code = clip_model.encode_text(text_token).to(dtype=torch.float32).cuda()

                if 'gt' in current_path:
                    gt = torch.from_numpy(img_as_float32(Y_gt)).cuda()
                else:
                    gt = None
                input_Z = torch.from_numpy(Z).cuda()

                # EM-algorithm
                for _ in range(self.max_iter_EM):
                    # M-Step　
                    self.optimizerD.zero_grad()
                    optimizerG.zero_grad()
                    # rain_gen_M, state_next = self.G_forward_truncate(input_Z, input_S, input_M)
                    # rain_gen_M, state_next = self.G_forward_truncate(input_Z, input_M)
                    rain_gen_M, state_next = self.G_forward_truncate(input_Z, text_code, input_M)

                    back_pre = self.RestorationNet(inputs, text_code)
                    lossM, likelihood, mse_scale, tv, criter, ssim_scale = self.get_loss_MRStep(inputs, back_pre, rain_gen_M, gt)
                    lossM.backward()
                    current_norm_grad_D = nn.utils.clip_grad_norm_(self.RestorationNet.parameters(), self.max_grad_norm_D)
                    # current_norm_grad_D = torch.nn.utils.clip_grad_norm_(self.DNet.parameters(), 0.01, error_if_nonfinite=False)
                    self.optimizerD.step()
                    if (ii + 1) > self.pretrain:
                        optimizerG.step()

                    # accumulate loss of M-Step
                    lossM_batch += lossM.item()
                    likelihood_batch += likelihood.item()
                    mse_batch += mse_scale.item()
                    tv_batch += tv.item()
                    criter_batch += criter.item()
                    ssim_batch += ssim_scale.item()
                    mean_norm_grad_batch_D += current_norm_grad_D

                    # E-Step　
                    if (ii + 1) > self.pretrain:
                        self.freeze_Generator()
                        rain_gt = inputs - back_pre.detach()
                        for ss in range(self.langevin_steps):
                            input_Z.requires_grad = True
                            input_M.requires_grad = True

                            # rain_gen_E, state_next = self.G_forward_truncate(input_Z, input_M)
                            rain_gen_E, state_next = self.G_forward_truncate(input_Z, text_code, input_M)
                            # rain_gen_E, _ = self.G_forward_truncate(input_Z, text_code)
                            lossE = self.get_loss_EStep(rain_gt, rain_gen_E)
                            lossE.backward()

                            input_Z = input_Z - 0.5 * (self.delta ** 2) * (input_Z.grad + input_Z / (num_batch))
                            # input_M = input_M - 0.5 * (self.delta ** 2) * (input_M.grad + input_M / (num_batch * num_frame))
                            input_M = input_M - 0.5 * (self.delta ** 2) * (input_M.grad + input_M / (num_batch))
                            if ss < (self.langevin_steps / 3):
                                input_Z = input_Z + self.delta * torch.randn_like(input_Z)
                                input_M = input_M + self.delta * torch.randn_like(input_M)
                            input_Z.detach_()
                            input_M.detach_()
                        self.unfreeze_Generator()

                # update Z_rank and S_rank
                if (ii + 1) > self.pretrain:
                    Z = input_Z.data.cpu().numpy()
                    M = input_M.data.cpu().numpy()

                # tensorboard
                if random.randint(1, 20) == 1:
                    ind_batch = random.randint(0, rain_gen_M.shape[0] - 1)
                    x1 = vutils.make_grid(inputs[ind_batch,].squeeze(), normalize=False, scale_each=False)
                    # x1 = vutils.make_grid(inputs[ind_batch,].squeeze().permute([1, 0, 2, 3]), normalize=False, scale_each=False)
                    self.writer.add_image('Train Rainy Image', x1, self.log_im_step['train'])
                    x3 = vutils.make_grid(back_pre[ind_batch,].squeeze().clamp_(0.0, 1.0), normalize=False,
                                          scale_each=False)
                    # x3 = vutils.make_grid(back_pre[ind_batch,].squeeze().permute([1, 0, 2, 3]).clamp_(0.0, 1.0), normalize=False, scale_each=False)
                    self.writer.add_image('Train Deained Image', x3, self.log_im_step['train'])
                    x4 = rain_gen_M[ind_batch,].squeeze()
                    # x4 = rain_gen_M[ind_batch,].squeeze().permute([1, 0, 2, 3])
                    x5 = (inputs[ind_batch,] - back_pre[ind_batch,]).squeeze().clamp_(min=0)
                    # x5 = (inputs[ind_batch,] - back_pre[ind_batch,]).squeeze().permute([1, 0, 2, 3]).clamp_(min=0)
                    temp = vutils.make_grid(torch.stack((x4, x5), dim=0), normalize=True, scale_each=True)
                    # temp = vutils.make_grid(torch.cat([x4, x5], dim=0), normalize=True, scale_each=True)
                    self.writer.add_image('Train Rains and Residual', temp, self.log_im_step['train'])
                    self.log_im_step['train'] += 1

                # save the updated latent variable
                if (ii + 1) > self.pretrain:
                    savemat(current_path['latent'], {'Z': Z})
                    savemat(current_path['state'], {'S': S})
                    savemat(current_path['motion'], {'M': M})

                # calculate the mean loss of each video
                lossM_batch /= self.max_iter_EM
                likelihood_batch /= self.max_iter_EM
                mse_batch /= self.max_iter_EM
                tv_batch /= self.max_iter_EM
                criter_batch /= self.max_iter_EM
                ssim_batch /= self.max_iter_EM
                mean_norm_grad_batch_D /= self.max_iter_EM

                lossM_epoch += lossM_batch
                likelihood_epoch += likelihood_batch
                mse_epoch += mse_batch
                tv_epoch += tv_batch
                criter_epoch += criter_batch
                ssim_epoch += ssim_batch
                mean_norm_grad_epoch_D += mean_norm_grad_batch_D

                # print log
                if (jj + 1) % self.print_freq == 0:
                    self.writer.add_scalar('LossM_Batch', lossM_batch, self.log_loss_step)
                    self.log_loss_step += 1

                    lr_D = self.optimizerD.param_groups[0]['lr']
                    lr_GState = optimizerG.param_groups[0]['lr']
                    lr_GRain = optimizerG.param_groups[1]['lr']

                    log_str = 'M-Step: Epoch:{:03d}/{:03d}, Video:{:03d}/{:03d}, ' + \
                              'LossM:{:.2e}({:.2e}/{:.2e}/{:.2e}/{:.2e}/{:.2e}), GradD:{:.2e}/{:.2e}, ' + \
                              'lrSRD:{:.2e}/{:.2e}/{:.2e}'
                    print(log_str.format(ii + 1, self.epochs, jj + 1, len(self.train_data_list),
                                         lossM_batch, likelihood_batch, mse_batch, tv_batch, criter_batch, ssim_batch,
                                         mean_norm_grad_batch_D,
                                         self.max_grad_norm_D, lr_GState, lr_GRain, lr_D))

                # save the rain generator
                if (ii+1) >= self.pretrain:
                    torch.save({'GState': self.GStateNet.state_dict(),
                                'GRain': self.GRainNet.state_dict()}, current_path['generator'])

            # calculate the mean loss of each epoch
            lossM_epoch /= (jj + 1)
            likelihood_epoch /= (jj + 1)
            mse_epoch /= (jj + 1)
            tv_epoch /= (jj + 1)
            criter_epoch /= (jj + 1)
            ssim_epoch /= (jj + 1)
            mean_norm_grad_epoch_D /= (jj + 1)

            # print loss and testing
            print('-' * 150)
            log_str = 'Train: Epoch:{:02d}/{:02d}, LossM:{:.2e} ({:.2e}/{:.2e}/{:.2e}/{:.2e}/{:.2e}), GradD:{:.2e}/{:.2e}'
            print(log_str.format(ii + 1, self.epochs, lossM_epoch, likelihood_epoch, mse_epoch, tv_epoch, criter_epoch, ssim_epoch,
                                 mean_norm_grad_epoch_D, self.max_grad_norm_D))

            # # testing
            self.predict_deraining(args)
            print('=' * 150)
            log_str = 'Testing RainDrop: Epoch:{:02d}/{:02d}, PSNR={:4.2f}, SSIM={:6.4f}'
            print(log_str.format(ii + 1, self.epochs, self.psnrm_raindrop, self.ssimm_raindrop))
            log_str = 'Testing RainHaze: Epoch:{:02d}/{:02d}, PSNR={:4.2f}, SSIM={:6.4f}'
            print(log_str.format(ii + 1, self.epochs, self.psnrm_rainhaze, self.ssimm_rainhaze))
            log_str = 'Testing Snow100k-S: Epoch:{:02d}/{:02d}, PSNR={:4.2f}, SSIM={:6.4f}'
            print(log_str.format(ii + 1, self.epochs, self.psnrm_snow_S, self.ssimm_snow_S))
            log_str = 'Testing Snow100k-L: Epoch:{:02d}/{:02d}, PSNR={:4.2f}, SSIM={:6.4f}'
            print(log_str.format(ii + 1, self.epochs, self.psnrm_snow_L, self.ssimm_snow_L))
            log_str = 'Testing Average: Epoch:{:02d}/{:02d}, PSNR={:4.2f}, SSIM={:6.4f}'
            print(log_str.format(ii + 1, self.epochs, self.psnrm_avg, self.ssimm_avg))
            print('=' * 150)

            # tensorboard
            self.writer.add_scalar('PSNR_raindrop', self.psnrm_raindrop, ii)
            self.writer.add_scalar('SSIM_raindrop', self.ssimm_raindrop, ii)
            self.writer.add_scalar('PSNR_rainhaze', self.psnrm_rainhaze, ii)
            self.writer.add_scalar('SSIM_rainhaze', self.ssimm_rainhaze, ii)
            self.writer.add_scalar('PSNR_Snow100K-S', self.psnrm_snow_S, ii)
            self.writer.add_scalar('SSIM_Snow100K-S', self.ssimm_snow_S, ii)
            self.writer.add_scalar('PSNR_Snow100K-L', self.psnrm_snow_L, ii)
            self.writer.add_scalar('SSIM_Snow100K-L', self.ssimm_snow_L, ii)
            self.writer.add_scalar('LossM_Epoch', lossM_epoch, ii)

            self.max_grad_norm_D = min(self.max_grad_norm_D, mean_norm_grad_epoch_D)

            # adjust learning rate
            self.decay_lr(ii)

            # save model for raindrop
            # model_prefix = 'model_raindrop'
            if self.psnrm_raindrop > best_psnr_raindrop and not math.isinf(self.psnrm_raindrop):
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)
                model_derain_prefix = 'model_best_psnr_raindrop'
                save_path_model_state = str(Path(self.model_dir) / (model_derain_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_psnr_raindrop_epoch = ii + 1
                best_psnr_raindrop = self.psnrm_raindrop
            if self.ssimm_raindrop > best_ssim_raindrop:
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)

                model_derain_prefix = 'model_best_ssim_raindrop'
                save_path_model_state = str(Path(self.model_dir) / (model_derain_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_ssim_raindrop_epoch = ii + 1
                best_ssim_raindrop = self.ssimm_raindrop

            # save model for rainhaze
            # model_prefix = 'model_rainhaze'
            if self.psnrm_rainhaze > best_psnr_rainhaze and not math.isinf(self.psnrm_rainhaze):
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)
                model_derain_prefix = 'model_best_psnr_rainhaze'
                save_path_model_state = str(Path(self.model_dir) / (model_derain_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_psnr_rainhaze_epoch = ii + 1
                best_psnr_rainhaze = self.psnrm_rainhaze
            if self.ssimm_rainhaze > best_ssim_rainhaze:
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)

                model_derain_prefix = 'model_best_ssim_rainhaze'
                save_path_model_state = str(Path(self.model_dir) / (model_derain_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_ssim_rainhaze_epoch = ii + 1
                best_ssim_rainhaze = self.ssimm_rainhaze

            # save model for snow_S
            # model_prefix = 'model_snow_S'
            if self.psnrm_snow_S > best_psnr_snow_S and not math.isinf(self.psnrm_snow_S):
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)
                model_snow_prefix = 'model_best_psnr_snow_S'
                save_path_model_state = str(Path(self.model_dir) / (model_snow_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_psnr_snow_S_epoch = ii + 1
                best_psnr_snow_S = self.psnrm_snow_S
            if self.ssimm_snow_S > best_ssim_snow_S:
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)

                model_snow_prefix = 'model_best_ssim_snow_S'
                save_path_model_state = str(Path(self.model_dir) / (model_snow_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_ssim_snow_S_epoch = ii + 1
                best_ssim_snow_S = self.ssimm_snow_S

            # save model for snow_L
            # model_prefix = 'model_snow_L'
            if self.psnrm_snow_L > best_psnr_snow_L and not math.isinf(self.psnrm_snow_L):
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)
                model_snow_prefix = 'model_best_psnr_snow_L'
                save_path_model_state = str(Path(self.model_dir) / (model_snow_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_psnr_snow_L_epoch = ii + 1
                best_psnr_snow_L = self.psnrm_snow_L
            if self.ssimm_snow_L > best_ssim_snow_L:
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)

                model_snow_prefix = 'model_best_ssim_snow_L'
                save_path_model_state = str(Path(self.model_dir) / (model_snow_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_ssim_snow_L_epoch = ii + 1
                best_ssim_snow_L = self.ssimm_snow_L

            # save model for allweather
            # model_prefix = 'model_allweather'
            if self.psnrm_avg > best_psnr_avg and not math.isinf(self.psnrm_avg):
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)
                model_snow_prefix = 'model_best_psnr_allweather'
                save_path_model_state = str(Path(self.model_dir) / (model_snow_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_psnr_avg_epoch = ii + 1
                best_psnr_avg = self.psnrm_avg
            if self.ssimm_avg > best_ssim_avg:
                # save_path_model = str(Path(self.model_dir) / (model_prefix + str(ii + 1)))
                # torch.save({'epoch': ii + 1,
                #             'step_loss': self.log_loss_step + 1,
                #             'step_img': {x: self.log_im_step[x] + 1 for x in self.log_im_step.keys()},
                #             'max_grad_norm_D': self.max_grad_norm_D,
                #             'DNet': self.RestorationNet.state_dict(),
                #             'optimizerD_state_dict': self.optimizerD.state_dict()}, save_path_model)

                model_snow_prefix = 'model_best_ssim_allweather'
                save_path_model_state = str(Path(self.model_dir) / (model_snow_prefix + str(ii + 1) + '.pt'))
                torch.save(self.RestorationNet.state_dict(), save_path_model_state)
                best_psnr_avg_epoch = ii + 1
                best_ssim_avg = self.ssimm_avg

        print('=' * 150)
        log_str = 'BEST_RainDrop_PSNR(epoch)={:4.2f}/{:02d}, BEST_RainDrop_SSIM(epoch):{:6.4f}/{:02d}'
        print(
            log_str.format(best_psnr_raindrop, best_psnr_raindrop_epoch, best_ssim_raindrop, best_ssim_raindrop_epoch))
        log_str = 'BEST_RainHaze_PSNR(epoch)={:4.2f}/{:02d}, BEST_RainHaze_SSIM(epoch):{:6.4f}/{:02d}'
        print(
            log_str.format(best_psnr_rainhaze, best_psnr_rainhaze_epoch, best_ssim_rainhaze, best_ssim_rainhaze_epoch))
        log_str = 'BEST_Snow100K_S_PSNR(epoch)={:4.2f}/{:02d}, BEST_Snow100K_S_SSIM(epoch):{:6.4f}/{:02d}'
        print(log_str.format(best_psnr_snow_S, best_psnr_snow_S_epoch, best_ssim_snow_S, best_ssim_snow_S_epoch))
        log_str = 'BEST_Snow100K_L_PSNR(epoch)={:4.2f}/{:02d}, BEST_Snow100K_L_SSIM(epoch):{:6.4f}/{:02d}'
        print(log_str.format(best_psnr_snow_L, best_psnr_snow_L_epoch, best_ssim_snow_L, best_ssim_snow_L_epoch))
        log_str = 'BEST_allweather_PSNR(epoch)={:4.2f}/{:02d}, BEST_allweather_SSIM(epoch):{:6.4f}/{:02d}'
        print(log_str.format(best_psnr_avg, best_psnr_avg_epoch, best_ssim_avg, best_ssim_avg_epoch))
        print('=' * 150)

        # close tensorboard
        self.writer.close()

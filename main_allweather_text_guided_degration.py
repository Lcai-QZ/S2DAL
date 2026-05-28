#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Power by Zongsheng Yue 2020-08-30 15:37:08

import commentjson as json
from trainers_singlegpu_allweather_text_guided_degration import trainer
from utils import str2none
import os
import torch
import argparse
from basicsr.utils.options import dict2str, parse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
torch.cuda.current_device()
torch.cuda._initialized = True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-opt', type=str, default='/media/zyserver/data16t/cailei/project_SAHistorFomer/SAHistoFormer_V2/Allweather/Options/Allweather_Histoformer_Degradation_Aware.yml', help='Path to option YAML file.')
    # set parameters
    with open('/media/zyserver/data16t/cailei/project_SAHistorFomer/SAHistoFormer_V2/options_allweather_text_guided_degration.json', 'r') as f:
    # with open('/mnt/disk/rfrf/S2VD-master/S2VD-master/options_derain_Rain200H.json', 'r') as f:
        args = json.load(f)
    args['resume'] = str2none(args['resume'])
    args['text_code_size'] = 512 ##
    yml = parser.parse_args()
    opt = parse(yml.opt)

    for key, value in args.items():
        print('{:<25s}: {:s}'.format(key, str(value)))

    # intialize the trainer
    # trainer_ntu = trainer(args)
    # if torch.cuda.is_available():
    #     torch.cuda.set_device(1)

    trainer_ntu = trainer(args, opt)

    # Begin training
    trainer_ntu.train(args, opt)

if __name__ == '__main__':
    main()
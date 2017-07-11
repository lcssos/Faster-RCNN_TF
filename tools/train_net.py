#!/usr/bin/env python
# -*- coding: utf-8 -*-

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""
# python ./tools/train_net.py --device cpu --device_id 0 --weights data/pretrain_model/VGG_imagenet.npy --imdb voc_2007_trainval --iters 70000 --cfg experiments/cfgs/faster_rcnn_end2end.yml --network VGGnet_train


import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
# import pdb

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    # 默认使用cpu进行训练，why？
    parser.add_argument('--device', dest='device', help='device to use',
                        default='cpu', type=str)
    parser.add_argument('--device_id', dest='device_id', help='device id to use',
                        default=0, type=int)
    parser.add_argument('--solver', dest='solver',
                        help='solver prototxt',
                        default=None, type=str)
    parser.add_argument('--iters', dest='max_iters',
                        help='number of iterations to train',
                        default=70000, type=int)
    # 预先训练的模型
    parser.add_argument('--weights', dest='pretrained_model',
                        help='initialize with pretrained model weights',
                        default=None, type=str)

    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)
    # 需要进行训练的数据集
    parser.add_argument('--imdb', dest='imdb_name',
                        help='dataset to train on',
                        default='kitti_train', type=str)
    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    # VGGnet_train
    parser.add_argument('--network', dest='network_name',
                        help='name of the network',
                        default=None, type=str)
    parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    print('\n./tools/train_net.py 显示命令行参数:')
    print(args)

    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('\n./tools/train_net.py 打印yaml配置文件')
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    print("\n./tools/train_net.py 开始加载训练数据集imdb：{}".format(args.imdb_name))
    imdb = get_imdb(args.imdb_name)
    # print(imdb.roidb)
    # print('Loaded dataset `{:s}` for training'.format(imdb.name))
    print("./tools/train_net.py 开始加载训练数据集roidb：")
    roidb = get_training_roidb(imdb)

    output_dir = get_output_dir(imdb, None)
    # print('Output will be saved to `{:s}`'.format(output_dir))
    print("\n./tools/train_net.py 输出文件路径`{:s}`".format(output_dir))

    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    # print device_name
    print("\n./tools/train_net.py  device_name:{}".format(device_name))

    network = get_network(args.network_name)
    # print 'Use network `{:s}` in training'.format(args.network_name)
    print("\n./tools/train_net.py 训练使用的神经网络 `{:s}`".format(args.network_name))
    # print("imdb.num_classes:{}".format(imdb.num_classes))

    train_net(network, imdb, roidb, output_dir,pretrained_model=args.pretrained_model,max_iters=args.max_iters)

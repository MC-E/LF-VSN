'''
Vimeo90K dataset
support reading images from lmdb, image folder and memcached
'''
import logging
import os
import os.path as osp
import pickle
import random

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

import data.util as util

try:
    import mc  # import memcached
except ImportError:
    pass
logger = logging.getLogger('base')

class Vimeo90KDataset(data.Dataset):
    '''
    Reading the training Vimeo90K dataset
    key example: 00001_0001 (_1, ..., _7)
    GT (Ground-Truth): 4th frame;
    LQ (Low-Quality): support reading N LQ frames, N = 1, 3, 5, 7 centered with 4th frame
    '''

    def __init__(self, opt):
        super(Vimeo90KDataset, self).__init__()
        self.opt = opt
        # get train indexes
        self.data_path = self.opt['data_path']
        self.txt_path = self.opt['txt_path']
        with open(self.txt_path) as f:
            self.list_video = f.readlines()
        self.list_video = [line.strip('\n') for line in self.list_video]
        # temporal augmentation
        self.interval_list = opt['interval_list']
        self.random_reverse = opt['random_reverse']
        logger.info('Temporal augmentation interval list: [{}], with random reverse is {}.'.format(
            ','.join(str(x) for x in opt['interval_list']), self.random_reverse))
        self.data_type = self.opt['data_type']
        random.shuffle(self.list_video)
        self.LR_input = True
        self.num_video = self.opt['num_video']

    def _ensure_memcached(self):
        if self.mclient is None:
            # specify the config files
            server_list_config_file = None
            client_config_file = None
            self.mclient = mc.MemcachedClient.GetInstance(server_list_config_file,
                                                          client_config_file)

    def __getitem__(self, index):
        GT_size = self.opt['GT_size']
        video_name = self.list_video[index]
        path_frame = os.path.join(self.data_path, video_name)
        frames = []
        for im_name in os.listdir(path_frame):
            if im_name.endswith('.png'):
                frames.append(util.read_img(None, osp.join(path_frame, im_name)))
        list_index_h = []
        index_h = random.randint(0, len(self.list_video) - 1)
        list_index_h.append(index_h)
        for _ in range(self.num_video-1):
            index_h_i = random.randint(0, len(self.list_video) - 1)
            while index_h_i == index or index_h_i in list_index_h:
                index_h_i = random.randint(0, len(self.list_video) - 1)
            list_index_h.append(index_h_i)

        # random crop
        H, W, C = frames[0].shape
        rnd_h = random.randint(0, max(0, H - GT_size))
        rnd_w = random.randint(0, max(0, W - GT_size))
        frames = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in frames]
        # stack HQ images to NHWC, N is the frame number
        img_frames = np.stack(frames, axis=0)
        # BGR to RGB, HWC to CHW, numpy to tensor
        img_frames = img_frames[:, :, :, [2, 1, 0]]
        img_frames = torch.from_numpy(np.ascontiguousarray(np.transpose(img_frames, (0, 3, 1, 2)))).float()
        # process h_list
        list_h = []
        for index_h_i in list_index_h:
            video_name_h = self.list_video[index_h_i]
            path_frame_h = os.path.join(self.data_path, video_name_h)
            frames_h = []
            for im_name in os.listdir(path_frame_h):
                if im_name.endswith('.png'):
                    frames_h.append(util.read_img(None, osp.join(path_frame_h, im_name)))
            frames_h = [v[rnd_h:rnd_h + GT_size, rnd_w:rnd_w + GT_size, :] for v in frames_h]
            img_frames_h = np.stack(frames_h, axis=0)
            img_frames_h = img_frames_h[:, :, :, [2, 1, 0]]
            img_frames_h = torch.from_numpy(np.ascontiguousarray(np.transpose(img_frames_h, (0, 3, 1, 2)))).float()
            list_h.append(img_frames_h.clone())
        list_h = torch.stack(list_h, dim=0)

        return {'GT': img_frames, 'LQ': list_h}

    def __len__(self):
        return len(self.list_video)
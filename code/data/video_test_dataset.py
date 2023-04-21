import os
import os.path as osp
import torch
import torch.utils.data as data
import data.util as util

class VideoTestDataset(data.Dataset):
    """
    A video test dataset. Support:
    Vid4
    REDS4
    Vimeo90K-Test

    no need to prepare LMDB files
    """

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.half_N_frames = opt['N_frames'] // 2
        self.data_path = opt['data_path']
        self.txt_path = self.opt['txt_path']
        self.num_video = self.opt['num_video']
        with open(self.txt_path) as f:
            self.list_video = f.readlines()
        self.list_video = [line.strip('\n') for line in self.list_video]
        self.list_video.sort()
        self.list_video = self.list_video[:200]
        l = len(self.list_video) // (self.num_video + 1)
        self.video_list_gt = self.list_video[:l]
        self.video_list_lq = self.list_video[l:l * (self.num_video + 1)]

    def __getitem__(self, index):
        path_GT = self.video_list_gt[index]  

        img_GT = util.read_img_seq(os.path.join(self.data_path, path_GT))
        list_h = []
        for i in range(self.num_video):
            path_LQ = self.video_list_lq[index*self.num_video+i] 
            imgs_LQ = util.read_img_seq(os.path.join(self.data_path, path_LQ))
            list_h.append(imgs_LQ)
        list_h = torch.stack(list_h, dim=0)
        return {
                'LQ': list_h,
                'GT': img_GT
            }
    
    def __len__(self):
        return len(self.video_list_gt)  

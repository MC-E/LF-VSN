import logging
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

import models.networks as networks
import models.lr_scheduler as lr_scheduler
from .base_model import BaseModel
from models.modules.loss import ReconstructionLoss
from models.modules.Quantization import Quantization
from .modules.common import DWT,IWT

logger = logging.getLogger('base')
dwt=DWT()
iwt=IWT()


class Model_VSN(BaseModel):
    def __init__(self, opt):
        super(Model_VSN, self).__init__(opt)

        if opt['dist']:
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = -1  # non dist training

        self.gop = opt['gop']
        train_opt = opt['train']
        test_opt = opt['test']
        self.opt = opt
        self.train_opt = train_opt
        self.test_opt = test_opt
        self.opt_net = opt['network_G']
        self.center = self.gop // 2
        self.num_video = opt['num_video']
        self.idxx = 0

        self.netG = networks.define_G_v2(opt).to(self.device)
        if opt['dist']:
            self.netG = DistributedDataParallel(self.netG, device_ids=[torch.cuda.current_device()])
        else:
            self.netG = DataParallel(self.netG)
        # print network
        self.print_network()
        self.load()

        self.Quantization = Quantization()

        if self.is_train:
            self.netG.train()

            # loss
            self.Reconstruction_forw = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_forw'])
            self.Reconstruction_back = ReconstructionLoss(losstype=self.train_opt['pixel_criterion_back'])
            self.Reconstruction_center = ReconstructionLoss(losstype="center")

            # optimizers
            wd_G = train_opt['weight_decay_G'] if train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.netG.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    if self.rank <= 0:
                        logger.warning('Params [{:s}] will not optimize.'.format(k))
            self.optimizer_G = torch.optim.Adam(optim_params, lr=train_opt['lr_G'],
                                                weight_decay=wd_G,
                                                betas=(train_opt['beta1'], train_opt['beta2']))
            self.optimizers.append(self.optimizer_G)

            # schedulers
            if train_opt['lr_scheme'] == 'MultiStepLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.MultiStepLR_Restart(optimizer, train_opt['lr_steps'],
                                                         restarts=train_opt['restarts'],
                                                         weights=train_opt['restart_weights'],
                                                         gamma=train_opt['lr_gamma'],
                                                         clear_state=train_opt['clear_state']))
            elif train_opt['lr_scheme'] == 'CosineAnnealingLR_Restart':
                for optimizer in self.optimizers:
                    self.schedulers.append(
                        lr_scheduler.CosineAnnealingLR_Restart(
                            optimizer, train_opt['T_period'], eta_min=train_opt['eta_min'],
                            restarts=train_opt['restarts'], weights=train_opt['restart_weights']))
            else:
                raise NotImplementedError('MultiStepLR learning rate scheme is enough.')

            self.log_dict = OrderedDict()

    def feed_data(self, data):
        self.ref_L = data['LQ'].to(self.device)  
        self.real_H = data['GT'].to(self.device)

    def init_hidden_state(self, z):
        b, c, h, w = z.shape
        h_t = []
        c_t = []
        for _ in range(self.opt_net['block_num_rbm']):
            h_t.append(torch.zeros([b, c, h, w]).cuda())
            c_t.append(torch.zeros([b, c, h, w]).cuda())
        memory = torch.zeros([b, c, h, w]).cuda()

        return h_t, c_t, memory

    def loss_forward(self, out, y):
        if self.opt['model'] == 'LSTM-VRN':
            l_forw_fit = self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out, y)
            return l_forw_fit
        elif self.opt['model'] == 'MIMO-VRN-h':
            l_forw_fit = 0
            for i in range(out.shape[1]):
                l_forw_fit += self.train_opt['lambda_fit_forw'] * self.Reconstruction_forw(out[:, i], y[:, i])
            return l_forw_fit

    def loss_back_rec(self, out, x):
        if self.opt['model'] == 'LSTM-VRN':
            l_back_rec = self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out, x)
            return l_back_rec
        elif self.opt['model'] == 'MIMO-VRN-h':
            l_back_rec = 0
            for i in range(x.shape[1]):
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out[:, i], x[:, i])
            return l_back_rec
    
    def loss_back_rec_mul(self, out, x):
        out = torch.chunk(out,self.num_video,dim=1)
        out = [outi.squeeze(1) for outi in out]
        x = torch.chunk(x,self.num_video,dim=1)
        x = [xi.squeeze(1) for xi in x]
        l_back_rec = 0
        for i in range(len(x)):
            for j in range(x[i].shape[1]):
                l_back_rec += self.train_opt['lambda_rec_back'] * self.Reconstruction_back(out[i][:, j], x[i][:, j])
        return l_back_rec

    def loss_center(self, out, x):
        # x.shape: (b, t, c, h, w)
        b, t = x.shape[:2]
        l_center = 0
        for i in range(b):
            mse_s = self.Reconstruction_center(out[i], x[i])
            mse_mean = torch.mean(mse_s)
            for j in range(t):
                l_center += torch.sqrt((mse_s[j] - mse_mean.detach()) ** 2 + 1e-18)
        l_center = self.train_opt['lambda_center'] * l_center / b

        return l_center

    def optimize_parameters(self, current_step):
        self.optimizer_G.zero_grad()
        
        b, n, t, c, h, w = self.ref_L.shape
        center = t // 2
        intval = self.gop // 2

        self.host = self.real_H[:, center - intval:center + intval + 1]
        self.secret = self.ref_L[:, :, center - intval:center + intval + 1]
        self.secret = [dwt(self.secret[:,i].reshape(b, -1, h, w)) for i in range(n)]
        self.output, out_h = self.netG(x=dwt(self.host.reshape(b, -1, h, w)), x_h=self.secret)
        self.output = iwt(self.output)

        Gt_ref = self.real_H[:, center - intval:center + intval + 1].detach()
        container = self.output[:, :3 * self.gop, :, :].reshape(-1, self.gop, 3, h, w)[:,self.gop//2]
        l_forw_fit = self.loss_forward(container.unsqueeze(1), Gt_ref[:,self.gop//2].unsqueeze(1))

        y = self.Quantization(self.output[:, :3 * self.gop, :, :].view(-1, self.gop, 3, h, w)[:,self.gop//2].unsqueeze(1).repeat(1,self.gop,1,1,1).reshape(b, -1, h, w))
        out_x, out_x_h, out_z = self.netG(x=dwt(y), rev=True)
        out_x = iwt(out_x)
        out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]

        l_back_rec = self.loss_back_rec(out_x.reshape(-1, self.gop, 3, h, w)[:,self.gop//2].unsqueeze(1), self.host[:,self.gop//2].unsqueeze(1))
        out_x_h = torch.stack(out_x_h, dim=1)

        l_center_x = 0
        for i in range(n):
            l_center_x += self.loss_back_rec(out_x_h.reshape(-1, n, self.gop, 3, h, w)[:, :, self.gop//2].unsqueeze(2)[:,i], self.ref_L[:, :, center - intval:center + intval + 1][:,:,self.gop//2].unsqueeze(2)[:, i])

        loss = l_forw_fit*2 + l_back_rec + l_center_x*4
        loss.backward()

        if self.train_opt['lambda_center'] != 0:
            self.log_dict['l_center_x'] = l_center_x.item()

        # set log
        self.log_dict['l_back_rec'] = l_back_rec.item()
        self.log_dict['l_forw_fit'] = l_forw_fit.item()
        
        self.log_dict['l_h'] = (l_center_x*10).item()

        # gradient clipping
        if self.train_opt['gradient_clipping']:
            nn.utils.clip_grad_norm_(self.netG.parameters(), self.train_opt['gradient_clipping'])

        self.optimizer_G.step()

    def test(self):
        self.netG.eval()
        with torch.no_grad():
            forw_L = []
            forw_L_h = []
            fake_H = []
            fake_H_h = []
            pred_z = []
            b, t, c, h, w = self.real_H.shape
            center = t // 2
            intval = self.gop // 2
            ids=[-1,0,1]
            b, n, t, c, h, w = self.ref_L.shape
            for j in range(3):
                id=ids[j]
                # forward downscaling
                self.host = self.real_H[:, center - intval+id:center + intval + 1+id]
                self.secret = self.ref_L[:, :, center - intval+id:center + intval + 1+id]
                self.secret = [dwt(self.secret[:,i].reshape(b, -1, h, w)) for i in range(n)]
                self.output, out_h = self.netG(x=dwt(self.host.reshape(b, -1, h, w)),x_h=self.secret)
                self.output = iwt(self.output)
                out_lrs = self.output[:, :3 * self.gop, :, :].reshape(-1, self.gop, 3, h, w)

                # backward upscaling
                y = self.Quantization(self.output[:, :3 * self.gop, :, :].view(-1, self.gop, 3, h, w)[:,self.gop//2].unsqueeze(1).repeat(1,self.gop,1,1,1).reshape(b, -1, h, w))
                out_x, out_x_h, out_z = self.netG(x=dwt(y), rev=True)
                out_x = iwt(out_x)
                out_x_h = [iwt(out_x_h_i) for out_x_h_i in out_x_h]
                out_x = out_x.reshape(-1, self.gop, 3, h, w)
                out_x_h = torch.stack(out_x_h, dim=1)
                out_x_h = out_x_h.reshape(-1, n, self.gop, 3, h, w)
                forw_L.append(out_lrs[:, self.gop//2])
                fake_H.append(out_x[:, self.gop//2])
                fake_H_h.append(out_x_h[:,:, self.gop//2])

        self.fake_H = torch.clamp(torch.stack(fake_H, dim=1),0,1)
        self.fake_H_h = torch.clamp(torch.stack(fake_H_h, dim=2),0,1)
        self.forw_L = torch.clamp(torch.stack(forw_L, dim=1),0,1)
        self.netG.train()
        
    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self):
        b, n, t, c, h, w = self.ref_L.shape
        center = t // 2
        intval = 3 // 2
        out_dict = OrderedDict()
        LR_ref = self.ref_L[:, :, center - intval:center + intval + 1].detach()[0].float().cpu()
        LR_ref = torch.chunk(LR_ref, self.num_video, dim=0)
        out_dict['LR_ref'] = [video.squeeze(0) for video in LR_ref]
        out_dict['SR'] = self.fake_H.detach()[0].float().cpu()
        SR_h = self.fake_H_h.detach()[0].float().cpu()
        SR_h = torch.chunk(SR_h, self.num_video, dim=0)
        out_dict['SR_h'] = [video.squeeze(0) for video in SR_h]
        out_dict['LR'] = self.forw_L.detach()[0].float().cpu()
        out_dict['GT'] = self.real_H[:, center - intval:center + intval + 1].detach()[0].float().cpu()

        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel) or isinstance(self.netG, DistributedDataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)
        if self.rank <= 0:
            logger.info('Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
            logger.info(s)

    def load(self):
        load_path_G = self.opt['path']['pretrain_model_G']
        if load_path_G is not None:
            logger.info('Loading model for G [{:s}] ...'.format(load_path_G))
            self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])
    
    def load_test(self,load_path_G):
        self.load_network(load_path_G, self.netG, self.opt['path']['strict_load'])

    def save(self, iter_label):
        self.save_network(self.netG, 'G', iter_label)

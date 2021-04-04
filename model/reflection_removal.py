import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import scipy.io as sio
import os
from torch.utils.data import Dataset, DataLoader
import platform
import cv2
from . import vgg
from argparse import ArgumentParser
from skimage.measure import compare_ssim, compare_psnr
from .networks import SSIMLoss

class EdgeMap(nn.Module):
    def __init__(self, scale=1):
        super(EdgeMap, self).__init__()
        self.scale = scale
        self.requires_grad = False

    def forward(self, img):
        img = img / self.scale

        N, C, H, W = img.shape
        gradX = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)
        gradY = torch.zeros(N, 1, H, W, dtype=img.dtype, device=img.device)

        gradx = (img[..., 1:, :] - img[..., :-1, :]).abs().sum(dim=1, keepdim=True)
        grady = (img[..., 1:] - img[..., :-1]).abs().sum(dim=1, keepdim=True)

        gradX[..., :-1, :] += gradx
        gradX[..., 1:, :] += gradx
        gradX[..., 1:-1, :] /= 2

        gradY[..., :-1] += grady
        gradY[..., 1:] += grady
        gradY[..., 1:-1] /= 2

        # edge = (gradX + gradY) / 2
        edge = (gradX + gradY)

        return edge
class ReflectionRemovalModel(BaseModel):
    def name(self):
        return 'ReflectionRemovalModel'

    def initialize(self,opt):
        BaseModel.initialize(self, opt)
        # load/define networks
        self.psnr = []
        self.ssim = []
        self.p_sum = 0.0
        self.s_sum = 0.0
        self.gpu_ids = opt.gpu_ids
        self.netG = networks.define_G(self.gpu_ids)
        self.syn = networks.SynData(self.device)
        self.k_sz = np.linspace(opt.batchSize, 5, 80)  # for synthetic images
        self.lamda_dT = opt.lamda_dT
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.edge_map = EdgeMap(scale=1).to(self.device)
        self.lamda_dR = opt.lamda_dR
        self.lamda_cI = opt.lamda_cI
        self.lamda_I = opt.lamda_I
        self.lamda_T = opt.lamda_T
        self.lamda_T_edge = opt.lamda_T_edge
        self.vgg = vgg.Vgg19(requires_grad=False).to(self.device)
        self.trainFlag = True
        self.criterionIdt = torch.nn.MSELoss()
        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netG, 'G', which_epoch)

        if self.isTrain:
            # define loss functions
            torch.nn.utils.clip_grad_norm_(self.netG.parameters(), 0.25)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.ssimloss = SSIMLoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.criterionVgg = networks.VGGLoss1(self.device, vgg=self.vgg, normalize=False)
            self.criterionGradient = torch.nn.L1Loss().to(self.device)
            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        with torch.no_grad():
            if self.isTrain:
                if input['isNatural'][0] == 1:
                    self.isNatural = True
                else:
                    self.isNatural = False
                if not self.isNatural:  # Skip these procedures, if the data is from real-world.
                    T = input['T'].to(self.device)
                    R = input['R'].to(self.device)
                    if torch.mean(T) * 1 / 2 > torch.mean(R):
                        self.trainFlag = False
                        return
                    _, R, I, alpha = self.syn(T, R, self.k_sz)  # Synthesize data
                    self.alpha = round(alpha, 1)
                    if T.max() < 0.15 or R.max() < 0.15 or I.max() < 0.1:
                        self.trainFlag = False
                        return
                else:
                    I = input['I']
                    T = input['T']
            else:  # Test
                self.image_paths = input['B_paths']
                I = input['I']
                T = input['T']
        self.real_T = T.to(self.device)
        self.real_I = I.to(self.device)
#        self.fake_R = torch.ones_like(self.real_I) * 0.1
    def forward(self):
        self.Ts = []
        self.Rs = []
        self.V_Ts = []
        self.V_Rs = []
        self.S_Ts = []
        self.S_Rs = []
        self.e_Ts = []
        self.e_Rs = []
        T0 = self.real_I
        self.Ts.append(T0)
        R0 = torch.ones_like(self.real_I)
        self.Rs.append(R0)
    def test(self):
        self.forward()
        i = 0
        while i <= 3:
            T, R, e_T, e_R, V_T, V_R, S_T, S_R = self.netG(self.real_I,self.Ts[-1],self.Rs[-1])
            self.Ts.append(T)
            self.Rs.append(R)
            self.V_Ts.append(V_T)
            self.V_Rs.append(V_R)
            self.S_Rs.append(S_R)
            self.S_Ts.append(S_T)
            self.e_Rs.append(e_R)
            self.e_Ts.append(e_T)
            
            i = i+1
        T_final = self.Ts[-1]
        R_final = self.Rs[-1]
        #V_T_final = self.V_Ts[-1]
        #V_R_final = self.V_Rs[-1]
        T_edge_final = self.edge_map(T_final)
        T_real_edge = self.edge_map(self.real_T)
        #I_fake = normolize_img(self.gpu_ids,I_fake,T_final,R_final)
        self.T_final = T_final.data
        self.R_final = R_final.data
        self.res =(torch.abs(T_final-self.real_T)).data
#        self.V_final_T = self.V_Ts[-1].data
#        self.V_final_R = self.V_Rs[-1].data
#        self.S_T = self.S_Ts[-1].data
#        self.S_R = self.S_Rs[-1].data
#        self.e_T = self.e_Ts[-1].data
#        self.e_R = self.e_Rs[-1].data
        self.rec_I = (T_final+R_final).data
#        self.V_T_final = V_T_final.data
#        self.V_R_final = V_R_final.data

#        self.T_edge_final = T_edge_final
    # get image paths

    def get_image_paths(self):
        return self.image_paths
    def backward_G(self):
        self.forward()
        i = 0
        while i <= 2:
            T, R, e_T, e_R, V_T, V_R ,_,_= self.netG(self.real_I,self.Ts[-1],self.Rs[-1])
            #T, R = self.netG(self.real_I,self.Ts[-1],self.Rs[-1])
            self.Ts.append(T)
            self.Rs.append(R)
            i = i+1
        T_final = self.Ts[-1]
        R_final = self.Rs[-1]
        self.loss_idt_T = 0.0          # L_pixel on T
        self.loss_idt_R = 0.0          # L_pixel on R
        self.loss_res = 0.0            # L_residual: residual reconstruction loss
        self.loss_MP = 0.0             # L_MP: multi-scale perceptual loss
        self.loss_ssim = 0.0
        sigma = 1
        real_I_r = torch.pow(self.real_I, 2.2)
        real_T_r = torch.pow(self.real_T, 2.2)
#        real_T_r = torch.pow(self.real_T, 2.2)
#        I_real_r = torch.pow(self.real_I, 2.2)
#        T_real_edge = self.edge_map(self.real_T)
#        real_blended_r = torch.pow(self.real_blended, 2.2)

#        V_T_final = V_Ts[-1]
#        V_R_final = V_Rs[-1]
#        V_T_edge_final = self.edge_map(V_T_final)
#        V_I_fake_r = torch.pow(self.real_T+V_R_final, 2.2)
#        V_T_final_r = torch.pow(V_T_final, 2.2)
        
        
#        S_T_final = S_Ts[-1]
#        S_R_final = S_Rs[-1]
#        S_T_edge_final = self.edge_map(S_T_final)
#        S_I_fake_r = torch.pow(S_T_final+S_R_final, 2.2)
#        S_T_final_r = torch.pow(S_T_final, 2.2)

#        T_final = Ts[-1]
#        R_final = Rs[-1]
#        T_edge_final = self.edge_map(T_final)
#        I_fake_r = torch.pow(self.real_T + R_final, 2.2)
#        T_final_r = torch.pow(T_final, 2.2)
        iter_num = len(self.Ts)
        for i in range(iter_num):
            if i > 0:
                T_r = torch.pow(self.Ts[i], 2.2)
                R_r = torch.pow(self.Rs[i], 2.2)
                self.loss_idt_T += self.criterionIdt(self.Ts[i], self.real_T) * np.power(0.85, iter_num - i)
                self.loss_ssim += self.ssimloss(self.Ts[i], self.real_T) * np.power(0.85, iter_num - i)
                if not self.isNatural:
                    self.loss_res += self.criterionIdt(real_I_r, (T_r + R_r))\
                                        * np.power(0.85, iter_num - i) * 10
        for i in range(iter_num):
            if i > 0:
                T_r = torch.pow(self.Ts[i], 2.2)
                
        self.loss_MP = self.criterionVgg(self.Ts[-1], self.real_T)
#        self.loss_G = self.criterionGAN(self.netD(T_final), True) * 0.01  # L_adv: adversarial loss
        self.loss_T = 4*self.loss_idt_T + 1*self.loss_res + self.loss_MP #+ self.loss_G
#        self.loss_ssim = self.ssimloss(self.Ts[-1],self.real_T);
        self.loss_total =  self.loss_T + 0.4*self.loss_ssim
        
#        loss_discrepancyT = self.criterionL2(T_final_r, real_T_r)#+self.criterionL2(S_T_final_r, real_T_r)
#        loss_discrepancyI = self.criterionL2(I_fake_r, I_real_r)#+self.criterionL2(S_I_fake_r, I_real_r)
#        loss_T_edge = self.criterionGradient(T_edge_final, T_real_edge)#+self.criterionGradient(S_T_edge_final, T_real_edge)
#        loss_MP = self.criterionVgg(T_final, self.real_T)#+self.criterionVgg(S_T_final, self.real_T)
#        loss_discrepancyT = self.lamda_dT*loss_discrepancyT
#        loss_discrepancyI = self.lamda_dR*loss_discrepancyI
#        loss_T_edge = self.lamda_T_edge*loss_T_edge
#        loss_total = loss_discrepancyT + loss_discrepancyI + loss_MP+ loss_T_edge
        
        self.loss_total.backward()
      
        self.T_final = T_final.data
        self.R_final = R_final.data
#        self.T_edge_final = T_edge_final.data
#        self.V_T_final = V_T_final.data
#        self.V_R_final = V_R_final.data
#        self.V_T_edge_final = V_T_edge_final.data
        
        self.loss_ssim = self.loss_ssim.item()
        self.loss_MP = self.loss_MP.item()
#        self.loss_T_edge = loss_T_edge.item()
#        self.loss_discrepancyT = loss_discrepancyT.item()
#        self.loss_discrepancyI = loss_discrepancyI.item()
        self.loss_T = self.loss_T.item()

        self.loss_total = self.loss_total.item()


    def optimize_parameters(self):
        # forward
        if not self.trainFlag:
            self.trainFlag = True
            return
        self.forward()
        # G
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        ret_errors = OrderedDict([('loss_total', self.loss_total),
                                  ('loss_T',self.loss_T),
                                  ('loss_MP', self.loss_MP),
                                  ('loss_SSIM', self.loss_ssim)])
        return ret_errors

    def get_current_visuals_train(self):
#        real_transmission = util.tensor2im(self.real_transmission)
#        real_reflection = util.tensor2im(self.real_reflection)
        real_blended = util.tensor2im(self.real_I)
        T_final = util.tensor2im(self.T_final)
        R_final = util.tensor2im(self.R_final)
        real_transmission = util.tensor2im(self.real_T)
#        synthetic_C = util.tensor2im(self.synthetic_C)

        ret_visuals = OrderedDict([('real_blended',real_blended),('T_final', T_final),('real_transmission',real_transmission), ('R_final', R_final)])
                                #    ('real_transmission', real_transmission), ('real_reflection', real_reflection), ('synthetic_C', synthetic_C)
        return ret_visuals

    def get_current_visuals_test(self):
        real_blended = util.tensor2im(self.real_I)
        T_final = util.tensor2im(self.T_final)
        R_final = util.tensor2im(self.R_final)
#        V_final_T = util.tensor2im(self.V_final_T)
#        V_final_R = util.tensor2im(self.V_final_R)
#        S_T = util.tensor2im(self.S_T)
#        S_R = util.tensor2im(self.S_R)
#        e_T = util.tensor2im(self.e_T)
#        e_R = util.tensor2im(self.e_R)
        T_real = util.tensor2im(self.real_T)
        res = util.tensor2im(self.res)
        rec_I = util.tensor2im(self.rec_I)
        
        T_final_gry = cv2.cvtColor(T_final, cv2.COLOR_BGR2GRAY)
        GT_gry = cv2.cvtColor(T_real, cv2.COLOR_BGR2GRAY)
        p = compare_psnr(T_final,T_real)
        s = compare_ssim(T_final_gry,GT_gry)
        self.psnr.append(p)
        self.ssim.append(s)
        lens = len(self.psnr)
        self.s_sum += s
        self.p_sum += p
        
        print("PSNR:   %f  PSNR:aver   %f"%(p,self.p_sum/lens))
        print("SSIM:   %f  SSIM:aver   %f"%(s,self.s_sum/lens))
#        V_T_final = util.tensor2im(self.V_T_final)
#        V_R_final = util.tensor2im(self.V_R_final)
#        T_edge_final = util.tensor2im(self.T_edge_final)  ,('T_edge_final',T_edge_final)]             ,,('V_final_T',V_final_T),('V_final_R',V_final_R),('S_T',S_T),('S_R',S_R),('e_R',e_R),('e_T',e_T)
        real_transmission = util.tensor2im(self.real_T)
        ret_visuals = OrderedDict([('real_blended',real_blended),('T_final', T_final),('R_final', R_final),('real_transmission',real_transmission),('res',res),('recI',rec_I)])
#                                    ('fake_reflection', fake_reflection)
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

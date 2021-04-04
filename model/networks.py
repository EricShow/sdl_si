import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import math
import cv2
import scipy.stats as st
import code
from .vgg import Vgg19
from argparse import ArgumentParser
from model import blockNL
from model.DRnet import DRNet
###############################################################################
# Functions
###############################################################################

class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(self.window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size/2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).cuda())
        return window

    def ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1*mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel:
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            self.window = window
            self.channel = channel

        return 1 - self.ssim(img1, img2, window, self.window_size, channel, self.size_average)
def normalize(input_img):
    max0 = torch.max(input_img)
    min0 = torch.min(input_img)
    d = max0 - min0
#    print(d)
    min1 = torch.ones_like(input_img)*min0
    output = (input_img-min1)/d
    return output
def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)
def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
def define_G(gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())
    netG = MoG_DUN(gpu_ids)    
#    netG = torch.nn.DataParallel(MoG_DUN(), device_ids=[0, 1, 2])
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
#    init_weights(netG, init_type=init_type)
    return netG
def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #print(net)
    print('Total number of parameters: %d' % num_params)
class MoG_DUN(nn.Module):
    def __init__(self,gpu_ids = []):#
        super(MoG_DUN, self).__init__()
        self.gpu_ids = gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        in_channels=3
        out_channels=3
        n_feats=128
        n_resblocks=13
        self.n_iter = 1
        self.DRNet_Ts = DRNet(in_channels, out_channels, n_feats, n_resblocks)
        self.DRNet_Rs = DRNet(in_channels, out_channels, n_feats, 12)
        channels=3
        fs=15
        self.NLBlockTs = blockNL.blockNL(channels, fs)
        self.NLBlockRs = blockNL.blockNL(channels, fs)
#        self.Nonlocal = blockNL(channels,fs)
#        self.NLBlock = nn.ModuleList([blockNL.blockNL(channels, fs) for _ in range(4)])
        self.uT = nn.Parameter(torch.tensor(0.5))
        self.etaT = nn.Parameter(torch.tensor(0.5))
        self.gamaT = nn.Parameter(torch.tensor(0.5))
        self.deltaT = nn.Parameter(torch.tensor(0.1))
        self.gamaT1 = nn.Parameter(torch.tensor(0.5))
        self.deltaT1 = nn.Parameter(torch.tensor(0.1))
        self.uT1 = nn.Parameter(torch.tensor(0.5))

        self.uR = nn.Parameter(torch.tensor(0.5))
        self.etaR = nn.Parameter(torch.tensor(0.5))
        self.gamaR = nn.Parameter(torch.tensor(0.5))
        self.deltaR = nn.Parameter(torch.tensor(0.1))
        self.gamaR1 = nn.Parameter(torch.tensor(0.5))
        self.deltaR1 = nn.Parameter(torch.tensor(0.1))
        self.uR1 = nn.Parameter(torch.tensor(0.5))
        
        self.sigmoid = nn.Sigmoid()
    def forward(self, I, T, R):

        V_T = self.DRNet_Ts(T)
        S_T = self.NLBlockTs(T)
        V_R = self.DRNet_Rs(R)
        S_R = self.NLBlockRs(R)

        e_T = S_T - T
        e_R = S_R - R
#        print(e_R)
#        print(self.uT1)
#        print(self.gamaT1)
        e_T = e_T - self.deltaT1 * (self.uT1 * (I - T - e_T - R - e_R) * (-1) + self.gamaT1 * (T + e_T - S_T))

        e_R = S_R - R
        e_R = e_R - self.deltaR1 * (self.uR1 * (I - T - e_T - R - e_R) * (-1) + self.gamaR1 * (R + e_R - S_R))
        
#        T = T-self.deltaT*((T+R-I)+self.uT*(I-T-R)*(-1)+self.gamaT*(I-T-e_T-R-e_R)*(-1)+self.etaT*(T-V_T))
#        T = normalize(T)
#        R = R-self.deltaR*((T+R-I)+self.uR*(I-T-R)*(-1)+self.gamaR*(I-T-e_T-R-e_R)*(-1)+self.etaR*(R-V_R))
#        R = normalize(R)

        T = T-self.deltaT*((T+R-I)+self.uT*(I-T-e_T-R-e_R)*(-1)+self.gamaT*(S_T-T-e_T)*(-1)+self.etaT*(T-V_T))
        T = normalize(T)
        #T=self.sigmoid(T)
        R = R-self.deltaR*((T+R-I)+self.uR*(I-T-e_T-R-e_R)*(-1)+self.gamaR*(S_R-R-e_R)*(-1)+self.etaR*(R-V_R))
        R = normalize(R)
        #R=self.sigmoid(R)
        return T, R, e_T, e_R, V_T, V_R,S_T,S_R

class MeanShift(nn.Conv2d):
    def __init__(self, data_mean, data_std, data_range=1, norm=True):
        """norm (bool): normalize/denormalize the stats"""
        c = len(data_mean)
        super(MeanShift, self).__init__(c, c, kernel_size=1)
        std = torch.Tensor(data_std)
        self.weight.data = torch.eye(c).view(c, c, 1, 1)
        if norm:
            self.weight.data.div_(std.view(c, 1, 1, 1))
            self.bias.data = -1 * data_range * torch.Tensor(data_mean)
            self.bias.data.div_(std)
        else:
            self.weight.data.mul_(std.view(c, 1, 1, 1))
            self.bias.data = data_range * torch.Tensor(data_mean)
        self.requires_grad = False

class VGGLoss(nn.Module):
    def __init__(self, device, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6]
        self.indices = indices or [2]
        self.device = device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(self.device)
        else:
            self.normalize = None

    def __call__(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss


class VGGLoss1(nn.Module):
    def __init__(self, device, vgg=None, weights=None, indices=None, normalize=True):
        super(VGGLoss1, self).__init__()
        if vgg is None:
            self.vgg = Vgg19().cuda()
        else:
            self.vgg = vgg
        self.criterion = nn.L1Loss()
        self.weights = weights or [1.0 / 2.6, 1.0 / 4.8]
        self.indices = indices or [2, 7]
        self.device = device
        if normalize:
            self.normalize = MeanShift([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], norm=True).to(self.device)
        else:
            self.normalize = None
        print("Vgg: Weights: ", self.weights, " indices: ", self.indices, " normalize: ", self.normalize)

    def __call__(self, x, y):
        if self.normalize is not None:
            x = self.normalize(x)
            y = self.normalize(y)
        x_vgg, y_vgg = self.vgg(x, self.indices), self.vgg(y, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())

        return loss

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
def get_gaussian_kernel(kernel_size=3, sigma=2, channels=3):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    paddingl = (kernel_size - 1) // 2
    paddingr = kernel_size - 1 - paddingl
    pad = torch.nn.ReflectionPad2d((paddingl, paddingr, paddingl, paddingr))
    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels,
                                kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return nn.Sequential(pad, gaussian_filter)
def syn_data_Fan(t, r, sigma):
    sz = int(2 * np.ceil(2 * sigma) + 1)
    r = r.squeeze().numpy().transpose(1, 2, 0)
    r_blur = cv2.GaussianBlur(r, (sz, sz), sigma, sigma, 0)
    r_blur = torch.from_numpy(r_blur.transpose(2, 0, 1)).unsqueeze_(0).float()
    blend = r_blur + t
    if torch.max(blend) > 1:
        mean = torch.mean(blend[blend > 1])
        r_blur = r_blur - 1.3 * (mean - 1)
        r_blur = torch.clamp(r_blur, min=0, max=1)
        blend = torch.clamp(t + r_blur, min=0, max=1)

        if torch.max(blend) > 1:
            mean = torch.mean(blend[blend > 1])
            r_blur = r_blur - 1.3 * (mean - 1)
            r_blur = torch.clamp(r_blur, min=0, max=1)
            blend = torch.clamp(t + r_blur, min=0, max=1)

        if torch.isnan(r_blur).any() or torch.isnan(blend).any():
            print('sigma = %f, sz = %d, mean = %f' % (sigma, sz, mean))
            code.interact(local=locals())

    return t, r_blur, blend
def gkern(kernlen=100, nsig=1):
    """Returns a 2D Gaussian kernel array."""
    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel = kernel / kernel.max()
    return kernel
g_mask = gkern(560, 3)
g_mask = np.dstack((g_mask, g_mask, g_mask))
class SynData:
    def __init__(self, device):
        self.g_mask = torch.tensor(g_mask.transpose(2, 0, 1)).to(device)
        self.device = device

    def __call__(self, t: torch.Tensor, r: torch.Tensor, k_sz):
        device = self.device
        t = t.pow(2.2)
        r = r.pow(2.2)

        sigma = k_sz[np.random.randint(0, len(k_sz))]
        att = 1.08 + np.random.random() / 10.0
        alpha2 = 1 - np.random.random() / 5.0
        sz = int(2 * np.ceil(2 * sigma) + 1)
        g_kernel = get_gaussian_kernel(sz, sigma)
        g_kernel = g_kernel.to(device)
        r_blur: torch.Tensor = g_kernel(r).float()
        blend: torch.Tensor = r_blur + t

        maski = (blend > 1).float()
        mean_i = torch.clamp(torch.sum(blend * maski, dim=(2, 3)) / (torch.sum(maski, dim=(2, 3)) + 1e-6),
                             min=1).unsqueeze_(-1).unsqueeze_(-1)
        r_blur = r_blur - (mean_i - 1) * att
        r_blur = r_blur.clamp(min=0, max=1)

        h, w = r_blur.shape[2:4]
        neww = np.random.randint(0, 560 - w - 10)
        newh = np.random.randint(0, 560 - h - 10)
        alpha1 = self.g_mask[:, newh:newh + h, neww:neww + w].unsqueeze_(0)

        r_blur_mask = r_blur * alpha1
        blend = r_blur_mask + t * alpha2

        t = t.pow(1 / 2.2)
        r_blur_mask = r_blur_mask.pow(1 / 2.2)
        blend = blend.pow(1 / 2.2)
        blend = blend.clamp(min=0, max=1)

        return t, r_blur_mask, blend.float(), alpha2

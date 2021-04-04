import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, trainC, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=512, help='scale images to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='the basic number for channels in the network')
        self.parser.add_argument('--which_model_netG', type=str, default='ISTANETplus', help='selects model to use for netG')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--nThreads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints_removal', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--name', type=str, default='ISTA', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--serial_batches', action='store_false', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--load_size',type=int,default=256)
        self.parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--dataset_name', type=str, default='nature',help='dir:checkpoints/dataset_name')
        self.parser.add_argument('--LayerNo', type=int, default=5,help='numbers of iters')
        self.parser.add_argument('--preprocess', type=str, default='resize_and_crop',help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        self.parser.add_argument('--no_flip', action='store_true',help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--lamda_dT', type=float, default=5, help='hyper-parameter of loss_discrapencyT')
        self.parser.add_argument('--lamda_dR', type=float, default=1, help='hyper-parameter of loss_discrapencyR')
        self.parser.add_argument('--lamda_cI', type=float, default=0.1, help='hyper-parameter of loss_constraintI')
        self.parser.add_argument('--lamda_I', type=float, default=1, help='hyper-parameter of loss_I_ks')
        self.parser.add_argument('--lamda_T', type=float, default=2, help='hyper-parameter of loss_T_ks')
        self.parser.add_argument('--lamda_T_edge', type=float, default=2, help='hyper-parameter of loss_T_edge')
        self.parser.add_argument('--dataset_size', type=int, default=4000, help='train dataset size, 0 means the whole set.')
        
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.dataset_name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--results_dir', type=str, default='./results_removal/', help='saves results here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--how_many', type=int, default=1000, help='how many test images to run')
        self.parser.add_argument('--GTroot',type=str,default='/home/shaodongliang/code/SIRR_ISTA/dataset/GT_nature/',help = 'dataroot0fGroundTruth')
        self.parser.add_argument('--crop_test', action='store_true', help='crop to 256*256 to test')
        self.isTrain = False

import sys
sys.path.append('..')
from options.base_options import BaseOptions

class TrainOptions(BaseOptions):
    def __init__(self):
        self.parser = BaseOptions.initialize(self)
        self.parser.add_argument('--epochs',default=200,type=int,help='train epochs')
        self.parser.add_argument('--b1',type=float,default=0.5)
        self.parser.add_argument('--b2',type=float,default=0.999)
        # self.parser.add_argument('--train_path',default='/home/ziweil/zhangxian/data/HQ/train',type=str)
        self.parser.add_argument('--train_path', default='/home/zhangxian/data/Celeba_HQ/train', type=str)
        self.parser.add_argument('--lr',default=0.0005,type=float)
        self.parser.add_argument('--ngc',default=32,type=int)
        self.parser.add_argument('--in_ch',default=3,type=int)
        self.parser.add_argument('--out_ch',default=3,type=int)
        # self.parser.add_argument('--valid_path',default='/home/ziweil/zhangxian/data/HQ/valid',type=str,help='valid path')
        # self.parser.add_argument('--test_path',default='/home/ziweil/zhangxian/data/HQ/test',type=str,help='valid path')
        # self.parser.add_argument('--save_path',default='/home/ziweil/zhangxian/model/deblur/deblur-face')
        self.parser.add_argument('--valid_path',default='/home/zhangxian/data/Celeba_HQ/valid',type=str,help='valid path')
        self.parser.add_argument('--test_path',default='/home/zhangxian/data/Celeba_HQ/test',type=str,help='valid path')
        self.parser.add_argument('--save_path',default='/home/zhangxian/model/deblur/STENet')

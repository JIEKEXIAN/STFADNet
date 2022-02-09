import argparse
import os

class BaseOptions():
    # def __init__(self):
    #     # parser = argparse.ArgumentParser()

    def initialize(self):
        parser = argparse.ArgumentParser()
        # self.parser.add_argument('--model',type=str,default='ENet',help='name of the model type')
        parser.add_argument('--shuffle',action='store_true',help='if true, takes images serial')
        parser.add_argument('--batch_size',type=int,default=8,help='input batch size')
        parser.add_argument('--image_size',default=256,type=int,help='image size')
        parser.add_argument('--iters',default=5,type=int,help='every iter to print loss')
        parser.add_argument('--nThreads',default=8,type=int,help='threads for loading data')
        parser.add_argument('--init_type',type=str,default='xavier',help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--latent_num',type=int,default=512,help='the shape of the latent vector')
        return parser

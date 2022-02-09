from model.base_network import BaseNetwork
import torch.nn as nn
from model.utils import *
import torch.nn.functional as F




class SN_Discriminator(BaseNetwork):
    def __init__(self,ndf=64):
        super(SN_Discriminator,self).__init__()
        #(N,128,128,64)
        self.opti_block = OptimizedBlock(3,ndf)
        #(N,64,64,128)
        self.one_block = ResBlock(64*1,64*2)
        #(N,32,32,256)
        self.two_block = ResBlock(64*2,64*4)
        #(N,16,16,512)
        self.three_block = ResBlock(64*4,64*8,predict=True)
        #(N,8,8,512)
        self.four_block = ResBlock(64*8,64*8,predict=True)
        #(N,4,4,512)
        self.five_block = ResBlock(64*8,64*8,predict=True)
        #(N,2,2,512)
        self.six_block = ResBlock(64*8,64*8)
        self.linear = nn.Linear(2*2*512,1)

    def forward(self,input):
        x = self.opti_block(input)
        x = self.one_block(x)
        x = self.two_block(x)
        result_one,x = self.three_block(x)
        result_two,x = self.four_block(x)
        result_three,x = self.five_block(x)
        x = self.six_block(x)
        x = x.view(x.size(0),-1)
        out = self.linear(x)
        return result_one,result_two,result_three,out

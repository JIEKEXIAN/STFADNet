from model.base_network import *
import torch.nn.functional as F
from model.utils import *

class LinkNet(BaseNetwork):
    def __init__(self,use_lconv=False):
        super(LinkNet, self).__init__()
        self.in_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(7, 7), stride=1, padding=3),
            nn.ReLU(inplace=True)
        )
        self.encoder1 = Encoder(64, 128, 3, 2, 1, norm='IN')
        self.encoder2 = Encoder(128, 256, 3, 2, 1, norm='IN')
        self.encoder3 = Encoder(256, 512, 3, 2, 1, norm='IN')
        self.encoder4 = Encoder(512, 512, 3, 2, 1, norm='IN')
        self.encoder5 = Encoder(512, 512, 3, 2, 1, norm='IN')

        self.activation = nn.ReLU()

        self.decoder5 = Decoder(512, 512, 3, 2, 1,1)
        self.ste5 = STEBlock(512,512,3)
        self.decoder4 = Decoder(512, 512, 3, 2, 1,1)
        self.ste4 = STEBlock(512,512,3)
        self.decoder3 = Decoder(512,256,3,2,1,1)
        self.ste3 = STEBlock(256,256,3)
        self.decoder2 = Decoder(256,128,3,2,1,1)
        self.ste2 = STEBlock(128,128,3)
        self.decoder1 = Decoder(128,64,3,2,1,1)


        self.out_block = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=(3, 3), stride=1, padding=1),
            nn.Tanh())

    def forward(self,input,segmap):
        x = self.in_block(input)
        # e1:(batch,32,128,128)
        e1 = self.encoder1(x)
        # e2:(batch,64,64,64)
        e2 = self.encoder2(e1)
        # e3:(batch,128,32,32)
        e3 = self.encoder3(e2)
        # e4:(batch,256,16,16)
        e4 = self.encoder4(e3)
        # e5:(batch,512,8,8)
        e5 = self.encoder5(e4)

        d5 = self.decoder5(e5)
        # d5 = e4+d5
        d5, texture5, structure5 = self.ste5(e4,d5,segmap)
        d5 = self.activation(d5)
        # d4:(batch,128,32,32)
        d4 = self.decoder4(d5)
        # d4 = e3+d4
        d4, texture4, structure4 = self.ste4(e3,d4,segmap)
        d4 = self.activation(d4)
        # d3:(batch,64,64,64)
        d3 = self.decoder3(d4)
        # d3 = e2+d3
        d3, texture3, structure3 = self.ste3(e2,d3,segmap)
        d3 = self.activation(d3)
        # d2:(batch,32,128,128)
        d2 = self.decoder2(d3)
        # d2 = e1+d2
        d2, texture2, structure2 = self.ste2(e1,d2,segmap)
        d2 = self.activation(d2)
        # d2:(batch,16,256,256)
        d1 = self.decoder1(d2)
        output = self.out_block(d1)
        return output,structure5,structure4,structure3,structure2,texture5,texture4,texture3,texture2
import torch
import sys
sys.path.append("..")
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def getnormalization(norm,channel):
    if norm=="BN":
        normalization = nn.BatchNorm2d(channel)
    if norm == 'IN':
        normalization = nn.InstanceNorm2d(channel)
    return normalization


class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Encoder(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size,stride=1,padding=0,groups=1,norm=None,bias=False,inplace=False):
        super(Encoder,self).__init__()
        self.norm=norm
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.InstanceNorm2d(in_planes//4),
                                nn.ReLU(inplace=False),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.InstanceNorm2d(in_planes//4),
                                nn.ReLU(inplace=False))
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.InstanceNorm2d(out_planes),
                                nn.ReLU(inplace=False),)

    def forward(self, x):
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)
        return x


class STEBlock(nn.Module):
    def __init__(self,in_planes,out_planes,kernel_size):
        super(STEBlock, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.activation = nn.ReLU()
        self.param_free_norm = getnormalization('BN',out_planes)
        self.param_st = nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,stride=1,padding=1)
        self.param_te = nn.Conv2d(in_planes,out_planes,kernel_size=kernel_size,stride=1,padding=1)
        self.parm_fusion = nn.Conv2d(in_planes*2,out_planes,kernel_size=1,padding=0,stride=1)
        self.conv1 = nn.Conv2d(out_planes,out_planes,kernel_size=kernel_size,padding=1,stride=1)
        self.conv2 = nn.Conv2d(out_planes,out_planes,kernel_size=kernel_size,padding=1,stride=1)
        self.conv1_1 = nn.Conv2d(out_planes,3,kernel_size=1,padding=0,stride=1)
        self.conv1_s = nn.Conv2d(out_planes,3,kernel_size=1,padding=0,stride=1)
        self.conv1_fu = nn.Conv2d(out_planes*2,out_planes,kernel_size=1,padding=0,stride=1)
        self.ad = AD(out_planes)

    # def forward(self,fe,fd,segmap):
    #     F_ed = fe+fd
    #     feature1 = self.conv1(F_ed)
    #     F_text = self.conv2(feature1)
    #     texture = self.tanh(self.conv1_1(F_text))
    #     F_structure = self.activation(self.param_st(fe))
    #     structure = self.tanh(self.conv1_s(F_structure))
    #     feature2 = torch.cat([F_text,F_structure],dim=1)
    #     feature_fusion = self.conv1_fu(feature2)
    #     out = self.ad(feature_fusion,segmap)
    #     return out,texture,structure

    def forward(self,fe,fd,segmap):
        F_ed = fe + fd
        feature1 = self.activation(self.conv1(F_ed))
        F_text = self.activation(self.conv2(feature1))
        texture = self.tanh(self.conv1_1(F_text))
        F_structure = self.activation(self.param_st(fe))
        structure = self.tanh(self.conv1_s(F_structure))
        feature2 = torch.cat([F_text, F_structure], dim=1)
        feature_fusion = self.activation(self.conv1_fu(feature2))
        out = self.ad(feature_fusion, segmap)
        return out, texture, structure





class AD(nn.Module):
    def __init__(self,norm_nc,label_nc=3):
        super(AD,self).__init__()
        nhidden = 128
        self.param_free_norm = nn.InstanceNorm2d(norm_nc,affine=False)
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc,nhidden,kernel_size=3,padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden,norm_nc,kernel_size=3,padding=1)
        self.mlp_beta = nn.Conv2d(nhidden,norm_nc,kernel_size=3,padding=1)

    def forward(self,x,segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap,size=x.size()[2:],mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        out = normalized * (1+gamma) + beta
        return out



class OptimizedBlock(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(OptimizedBlock,self).__init__()
        self.activation=nn.ReLU()
        self.res_block = self.make_res_block(in_channels,out_channels)
        self.res_connect = self.make_residual_connect(in_channels,out_channels)

    def make_res_block(self,in_channels,out_channels):
        model = []
        model+=[spectral_norm(nn.Conv2d(in_channels,out_channels,3,1,1))]
        model+=[nn.ReLU()]
        model+=[spectral_norm(nn.Conv2d(out_channels,out_channels,4,2,1))]
        return nn.Sequential(*model)

    def make_residual_connect(self,in_channels,out_channels):
        model=[]
        model+=[spectral_norm(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1))]
        model+=[nn.ReLU()]
        model += [spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1))]
        return nn.Sequential(*model)

    def forward(self,input):
        out = self.res_block(input)+self.res_connect(input)
        return self.activation(out)

class ResBlock(nn.Module):
    def __init__(self,input_nc,output_nc,predict=False):
        super(ResBlock, self).__init__()
        self.activation = nn.ReLU()
        self.predict = predict
        self.resblock = self.make_res_block(input_nc,output_nc,input_nc)
        self.res_conncet = self.make_res_connect(input_nc,output_nc)
        if self.predict:
            self.P_conv = nn.Sequential(nn.Conv2d(output_nc,output_nc,3,1,1))

    def make_res_block(self,in_channels,out_channels,hidden_channels):
        model = []
        model+=[spectral_norm(nn.Conv2d(in_channels,hidden_channels,kernel_size=3,stride=1,padding=1))]
        model+=[nn.ReLU()]
        model+=[spectral_norm(nn.Conv2d(hidden_channels,out_channels,kernel_size=4,stride=2,padding=1))]
        return nn.Sequential(*model)

    def make_res_connect(self,in_channels,out_channels):
        model=[]
        model+=[spectral_norm(nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=1,padding=1))]
        model+=[nn.ReLU()]
        model+=[spectral_norm(nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1))]
        return nn.Sequential(*model)

    def forward(self,input):
        out = self.resblock(input)+self.res_conncet(input)
        out = self.activation(out)
        if self.predict:
            result = self.P_conv(out)
            return result,out
        return out
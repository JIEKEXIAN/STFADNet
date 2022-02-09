import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.optim as optim
from model.geneartor import LinkNet
from torch.autograd import Variable
from model.discriminator import SN_Discriminator
# from model.MRNNet import LinkNet
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
from model.function import tensor2img
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data.data import *
import wandb
from options.train_options import *
wandb.init(project="deblur-face")
Tensor = torch.cuda.FloatTensor
device = torch.device("cuda:0")
class Train():
    def __init__(self,opt,train_loader,valid_loader):
        self.opt = opt
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.init_network()

    def init_network(self):
        self.G = LinkNet()
        self.D = SN_Discriminator()
        self.G.init_weights()
        self.G.to(device)
        # self.G = nn.DataParallel(self.G)
        self.D.init_weights()
        self.D.to(device)
        # self.D = nn.DataParallel(self.D)

        self.crition_l1 = nn.L1Loss()
        self.crition_l1.cuda()
        self.BCE = nn.BCEWithLogitsLoss()
        self.BCE.cuda()
        # self.vgg16 = Vgg16(requires_grad=False)
        # self.vgg16.cuda()
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.opt.lr,betas=(0.5, 0.999))
        self.scheduler = lr_scheduler.StepLR(self.G_optimizer,step_size=20,gamma=0.8)
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.opt.lr,betas=(0.5, 0.999))

    def D_loss(self,D_real_one,D_real_two,D_real_three,D_real,D_fake_one,D_fake_two,D_fake_three,D_fake):
        loss = 0
        one_label = Variable(Tensor(D_real_one.size()).fill_(1.0),requires_grad=False)
        loss+=0.2*self.BCE(D_real_one-D_fake_one,one_label)

        two_label = Variable(Tensor(D_real_two.size()).fill_(1.0), requires_grad=False)
        loss += 0.4 * self.BCE(D_real_two - D_fake_two, two_label)
        three_label = Variable(Tensor(D_real_three.size()).fill_(1.0), requires_grad=False)
        loss += 0.8 * self.BCE(D_real_three - D_fake_three, three_label)
        label = Variable(Tensor(D_real.size()).fill_(1.0), requires_grad=False)
        loss = loss+self.BCE(D_real-D_fake,label)
        # wandb.log({"adv_D": loss})
        return loss

    # def G_loss(self,real_one,real_two,real_three,real,fake_one,fake_two,fake_three,fake):
    #     adv_loss += self.BCE(fake-real, label)
    #     # wandb.log({"adv_g_loss":adv_loss})
    #     #  wandb.log({"rec_loss":rec_loss.mean().item()})
    #     loss = 120*rec_loss+0.1*adv_loss
    #     return loss


    def rec_loss(self,out,target,skin,hair,back,facial):
        skinloss = 12 * self.crition_l1(out * skin, target * skin)
        facialloss = 10 * self.crition_l1(out * facial,target * facial)
        hairloss = 8 * self.crition_l1(out * hair, target * hair)
        backloss = 6 * self.crition_l1(out * back, target * back)
        loss = skinloss + facialloss + hairloss + backloss
        wandb.log({"rec_loss":loss.mean().item()})
        return loss

    def structue_texture_loss(self,structure5,structure4,structure3,structure2,texture5,texture4,texture3,texture2,structure_gt,texture_gt):
        structure_gt5 = F.interpolate(structure_gt,size=structure5.size()[2:],mode='bilinear')
        structure_gt4 = F.interpolate(structure_gt,size=structure4.size()[2:],mode='bilinear')
        structure_gt3 = F.interpolate(structure_gt,size=structure3.size()[2:],mode='bilinear')
        structure_gt2 = F.interpolate(structure_gt,size=structure2.size()[2:],mode='bilinear')
        texture_gt5 = F.interpolate(texture_gt,size=texture5.size()[2:],mode='bilinear')
        texture_gt4 = F.interpolate(texture_gt,size=texture4.size()[2:],mode='bilinear')
        texture_gt3 = F.interpolate(texture_gt,size=texture3.size()[2:],mode='bilinear')
        texture_gt2 = F.interpolate(texture_gt,size=texture2.size()[2:],mode='bilinear')
        structue_loss = self.crition_l1(structure5,structure_gt5) + self.crition_l1(structure4,structure_gt4)\
                        +self.crition_l1(structure3,structure_gt3)+ self.crition_l1(structure2,structure_gt2)
        texture_loss = self.crition_l1(texture5,texture_gt5)+self.crition_l1(texture4,texture_gt4)\
                       + self.crition_l1(texture3,texture_gt3)+ self.crition_l1(texture2,texture_gt2)
        return structue_loss+texture_loss

    def adv_loss(self,real_one,real_two,real_three,real,fake_one,fake_two,fake_three,fake):
        adv_loss = 0
        one_label = Variable(Tensor(real_one.size()).fill_(1.0), requires_grad=False)
        adv_loss += 0.2 * self.BCE(fake_one - real_one, one_label)
        two_label = Variable(Tensor(real_two.size()).fill_(1.0), requires_grad=False)
        adv_loss += 0.4 * self.BCE(fake_two - real_two, two_label)
        three_label = Variable(Tensor(real_three.size()).fill_(1.0), requires_grad=False)
        adv_loss += 0.8 * self.BCE(fake_three - real_three, three_label)
        label = Variable(Tensor(real.size()).fill_(1.0), requires_grad=False)
        adv_loss += self.BCE(fake - real, label)
        wandb.log({"g_adv_loss":adv_loss.mean().item()})
        return adv_loss



    def train(self):
        save_path = os.path.join(self.opt.save_path,'deblur_IN')
        model_path = os.path.join(save_path,'model')
        # wandb.log({"lr": self.G_optimizer.state_dict()['param_groups'][0]['lr']})
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        for epoch in range(self.opt.epochs):
            self.scheduler.step()
            self.G.train()
            # self.valid(epoch, save_path)
            for i,data in enumerate(self.train_loader):
                blur_img,target,skin,hair,facial,back,parsing,structure,img_name = data
                blur_img = blur_img.type(torch.cuda.FloatTensor)
                target = target.type(torch.cuda.FloatTensor)
                parsing = parsing.type(torch.cuda.FloatTensor)
                skin = skin.type(torch.cuda.FloatTensor)
                hair = hair.type(torch.cuda.FloatTensor)
                back = back.type(torch.cuda.FloatTensor)
                facial = facial.type(torch.cuda.FloatTensor)
                structure = structure.type(torch.cuda.FloatTensor)
                # real_ = Variable(Tensor(image.size(0), 1).fill_(1.0), requires_grad=False
                out,structure5,structure4,structure3,structure2,texture5,texture4,texture3,texture2= self.G(blur_img,parsing)
                D_real_one,D_real_two,D_real_three,D_real = self.D(target)
                D_fake_one,D_fake_two,D_fake_three,D_fake = self.D(out.detach())
                self.D_optimizer.zero_grad()
                d_loss = self.D_loss(D_real_one,D_real_two,D_real_three,D_real,D_fake_one,D_fake_two,D_fake_three,D_fake)
                wandb.log({"adv_loss":d_loss.mean().item()})
                d_loss.backward()
                self.D_optimizer.step()

                real_one,real_two,real_three,real = self.D(target)
                fake_one,fake_two,fake_three,fake = self.D(out)
                # feature_real = self.vgg16(target).relu4_3

                self.G_optimizer.zero_grad()
                # g_loss = self.G_loss(out,target,skin,hair,back,facial,real_one,real_two,real_three,real,fake_one,fake_two,fake_three,fake)
                loss1 = self.rec_loss(out,target,skin,hair,back,facial)
                loss2 = self.structue_texture_loss(structure5,structure4,structure3,structure2,texture5,texture4,texture3,texture2,structure,target)
                wandb.log({"fst_loss":loss2.mean().item()})
                loss3 = self.adv_loss(real_one,real_two,real_three,real,fake_one,fake_two,fake_three,fake)
                g_loss = 120*loss1 + loss2 + 0.1*loss3
                # seg_loss = self.seg_loss(segmap2,segmap3,segmap4,segmap5,parsing)
                g_loss.backward()
                self.G_optimizer.step()
                if i%5==0:
                    print("[%d/%d epochs],[%d/%d iters],[D_loss:%f],[g_loss:%f]"
                          % (epoch, self.opt.epochs, i, len(self.train_loader), d_loss.mean().item(),
                             g_loss.mean().item()))
            torch.save(self.G.state_dict(), os.path.join(model_path, str(epoch) + '_epoch_' + 'G_model.pkl'))

            self.G.eval()
            self.valid(epoch,save_path)

    def valid(self, epoch, save_path):
        face_path = os.path.join(save_path, str(epoch), 'face')
        if not os.path.exists(face_path):
            os.makedirs(face_path)
        # ppath2 = os.path.join(save_path,str(epoch),'par2')
        # ppath3 = os.path.join(save_path, str(epoch), 'par3')
        # ppath4 = os.path.join(save_path, str(epoch), 'par4')
        # ppath5 = os.path.join(save_path, str(epoch), 'par5')
        for i, data in enumerate(self.valid_loader):
            blur_img, parsing,img_name = data
            blur_img = blur_img.type(torch.cuda.FloatTensor)
            parsing = parsing.type(torch.cuda.FloatTensor)
            out,structure5,structure4,structure3,structure2,texture5,texture4,texture3,texture2 = self.G(blur_img, parsing)
            tensor2img(out.detach(), img_name, face_path)

if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = TrainOptions().parser.parse_args()
    img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    label_transform = transforms.Compose([transforms.ToTensor()])
    img_dir = os.listdir(os.path.join(opt.train_path, 'parsing'))
    traindata = Train_data(opt, img_dir, img_transform, label_transform)
    valid_dir = os.listdir(os.path.join(opt.valid_path, 'parsing'))
    validdata = Valid_data(opt, valid_dir, img_transform,label_transform)
    trainloader = DataLoader(traindata, batch_size=opt.batch_size, shuffle=opt.shuffle, num_workers=opt.nThreads)
    validloader = DataLoader(validdata, batch_size=4, num_workers=opt.nThreads)
    trainer = Train(opt, trainloader, validloader)
    trainer.train()





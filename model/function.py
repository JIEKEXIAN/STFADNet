import torch
from torchvision.utils import save_image
import os

def tensor2img(input,im_list,save_path):
    N,C,W,H = input.size()
    for i in range(N):
        filename = os.path.join(save_path,im_list[i])
        img = input[i]
        img = torch.unsqueeze(img,dim=0)
        save_image(img,filename=filename,nrow=1,padding=0,normalize=True)
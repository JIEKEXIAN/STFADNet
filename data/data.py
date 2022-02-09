from torch.utils.data import Dataset
from matplotlib.image import imread
from PIL import Image
import os

class Train_data(Dataset):
    def __init__(self,opt,img_dir,transform,transform_label):
        self.opt = opt
        self.train_path = opt.train_path
        self.transform = transform
        self.transform_label = transform_label
        self.img_dir = img_dir

    def __getitem__(self, item):
        blur_path = os.path.join(self.train_path,'blur_25',self.img_dir[item])
        image_path = os.path.join(self.train_path,'img',self.img_dir[item])
        skin_path = os.path.join(self.opt.train_path,'skin',self.img_dir[item])
        hair_path = os.path.join(self.opt.train_path,'hair',self.img_dir[item])
        facial_path = os.path.join(self.opt.train_path,'facial',self.img_dir[item])
        back_path = os.path.join(self.opt.train_path,'back',self.img_dir[item])
        parsing_path = os.path.join(self.opt.train_path,'parsing',self.img_dir[item])
        structure_path = os.path.join(self.opt.train_path,'str',self.img_dir[item])
        skin_mask = imread(skin_path)
        hair_mask = imread(hair_path)
        facial_mask = imread(facial_path)
        back_mask = imread(back_path)
        target = imread(image_path)
        structure = imread(structure_path)
        blur_img = imread(blur_path)
        blur_img = self.transform(blur_img)
        target = self.transform(target)
        # parsing = imread(parsing_path)
        parsing = Image.open(parsing_path).convert('RGB')
        parsing = self.transform(parsing)
        structure = self.transform(structure)
        skin_mask = self.transform_label(skin_mask)
        hair_mask = self.transform_label(hair_mask)
        facial_mask = self.transform_label(facial_mask)
        back_mask = self.transform_label(back_mask)
        return blur_img,target,skin_mask,hair_mask,facial_mask,back_mask,parsing,structure,self.img_dir[item]

    def __len__(self):
        return len(self.img_dir)


class Valid_data(Dataset):
    def __init__(self,opt,img_dir,transform,transform_label):
        self.opt = opt
        self.valid_path = opt.valid_path
        self.transform = transform
        self.transform_label = transform_label
        self.img_dir = img_dir

    def __getitem__(self, item):
        image_path = os.path.join(self.opt.valid_path,'blur_25',self.img_dir[item])
        parsing_path = os.path.join(self.opt.valid_path,'parsing',self.img_dir[item])
        blur_img = imread(image_path)
        # parsing = imread(parsing_path)
        parsing = Image.open(parsing_path).convert('RGB')
        blur_img = self.transform(blur_img)
        parsing = self.transform(parsing)
        return blur_img,parsing,self.img_dir[item]

    def __len__(self):
        return len(self.img_dir)


class Test_data(Dataset):
    def __init__(self,opt,img_dir,transform):
        self.opt = opt
        self.test_path = opt.test_path
        self.transform = transform
        self.img_dir = img_dir

    def __getitem__(self, item):
        image_path = os.path.join(self.test_path,'blur',self.img_dir[item])
        parsing_path = os.path.join(self.opt.test_path,'fine_25',self.img_dir[item])
        blur_img = imread(image_path)
        parsing = imread(parsing_path)
        blur_img = self.transform(blur_img)
        parsing = self.transform(parsing)
        return blur_img,parsing,self.img_dir[item]

    def __len__(self):
        return len(self.img_dir)
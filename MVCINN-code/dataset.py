import os
import random
import numpy as np
import torch
import cv2
import timm
import pandas as pd
import torchvision.transforms as transform
from sklearn import model_selection
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
class CustomDataset(Dataset):
    
    def __init__(self, df, data_path, transform=None):
        super().__init__()
        
        self.img_id = df['image'].values
        self.label = df['level'].values
        self.path = data_path
        self.transform = transform
    
    def __len__(self):
        return len(self.img_id)
    
    def __getitem__(self, idx):
        
        img_path = os.path.join(self.path, self.img_id[idx]+'.jpeg')
        assert os.path.exists(img_path), '{} img path is not exists...'.format(img_path)
        
        label = self.label[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label  # label不需要转换为tensor，在DataLoader中会通过collate_fn自动转换

class SingleimgDataset(torch.utils.data.Dataset):
    def __init__(self, imgs_dir,csv_list,transform = None):
        self.imgs_dir = imgs_dir
        self.csv_list = csv_list
        self.transform = transform
        #self.classes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011','012']    
        # imgs = os.listdir(imgs_dir)
        # for idx in range(len(self.data_list)):
        #     view_imgs = []
        #     for view in range(self.num_views):
        #         img_name = self.data_list.iloc[idx,view]+'.jpg'
        #         view_imgs.append(os.path.join(imgs_dir,img_name))
        #     self.imgs_path.append(view_imgs)
        # print("imgs_path:",len(self.imgs_path))

        # if self.test_mode:
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
          
        #     ])
        # else:
        #     self.transform = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])

        #     ])

    def __len__(self):
        return len(self.csv_list)

    def __getitem__(self, idx):
        class_id = self.csv_list.iloc[idx, 1].astype(np.int64)
        
        

        #classes = self.classes
        
        img_name = self.csv_list.iloc[idx,0]
        
        
        img_name = img_name if '.jpg' in img_name else img_name+'.jpg'
        img_path = os.path.join(self.imgs_dir,img_name)
        
        # im = Image.open(img_path).convert('RGB')
        # im = im.resize((224, 224), Image.ANTIALIAS)
        image = cv2.imread(img_path)
        if image is None:
            print("image error ",img_path,"is not exist!")
            raise ValueError("image error ",img_path,"is not exist!")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)  # 原生transform写法
            # image = self.transform(image=image)['image']
        

        #return class_id, torch.stack(imgs), path
        return  image,class_id

class MultiviewImgDataset(torch.utils.data.Dataset):

    def __init__(self, imgs_dir,data_list, scale_aug=False, rot_aug=False, test_mode=True, num_views=4,transform = None):
        self.imgs_dir = imgs_dir
        self.data_list = data_list
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.transform = transform
        #self.classes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011','012']    
        # imgs = os.listdir(imgs_dir)
        # for idx in range(len(self.data_list)):
        #     view_imgs = []
        #     for view in range(self.num_views):
        #         img_name = self.data_list.iloc[idx,view]+'.jpg'
        #         view_imgs.append(os.path.join(imgs_dir,img_name))
        #     self.imgs_path.append(view_imgs)
        # print("imgs_path:",len(self.imgs_path))

        # if self.test_mode:
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
          
        #     ])
        # else:
        #     self.transform = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])

        #     ])
        

    def __len__(self):
        return len(self.data_list)//4

    def __getitem__(self, idx):
        
        class_id = self.data_list.iloc[idx*4, 1].astype(np.int64)
        
        # Use PIL instead
        imgs = []

        #classes = self.classes
        for view in range(self.num_views):
            img_name = '{}.jpg'.format(self.data_list.iloc[idx*4+view,0])
            img_path = os.path.join(self.imgs_dir,img_name)
            # im = Image.open(img_path).convert('RGB')
            # im = im.resize((224, 224), Image.ANTIALIAS)
            image = cv2.imread(img_path)
            if image is None:
                print("image error ",img_path,"is not exist!")
                raise ValueError("image error ",img_path,"is not exist!")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)  # 原生transform写法
                # image = self.transform(image=image)['image']
            imgs.append(image)

        #return class_id, torch.stack(imgs), path
        # return  torch.stack(imgs),np.ones(4,dtype=np.int64)*class_id
        return  torch.stack(imgs),class_id

class MultiviewImgDataset_noAggregate(torch.utils.data.Dataset):

    def __init__(self, imgs_dir,data_list, scale_aug=False, rot_aug=False, test_mode=True, num_views=4,transform = None):
        self.imgs_dir = imgs_dir
        self.data_list = data_list
        self.scale_aug = scale_aug
        self.rot_aug = rot_aug
        self.test_mode = test_mode
        self.num_views = num_views
        self.transform = transform
        #self.classes = ['001', '002', '003', '004', '005', '006', '007', '008', '009', '010', '011','012']    
        # imgs = os.listdir(imgs_dir)
        # for idx in range(len(self.data_list)):
        #     view_imgs = []
        #     for view in range(self.num_views):
        #         img_name = self.data_list.iloc[idx,view]+'.jpg'
        #         view_imgs.append(os.path.join(imgs_dir,img_name))
        #     self.imgs_path.append(view_imgs)
        # print("imgs_path:",len(self.imgs_path))

        # if self.test_mode:
        #     self.transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])
          
        #     ])
        # else:
        #     self.transform = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                              std=[0.229, 0.224, 0.225])

        #     ])
        

    def __len__(self):
        return len(self.data_list)//4

    def __getitem__(self, idx):
        
        class_id = self.data_list.iloc[idx*4, 1].astype(np.int64)
        
        # Use PIL instead
        imgs = []

        #classes = self.classes
        for view in range(self.num_views):
            img_name = '{}.jpg'.format(self.data_list.iloc[idx*4+view,0])
            img_path = os.path.join(self.imgs_dir,img_name)
            # im = Image.open(img_path).convert('RGB')
            # im = im.resize((224, 224), Image.ANTIALIAS)
            image = cv2.imread(img_path)
            if image is None:
                print("image error ",img_path,"is not exist!")
                raise ValueError("image error ",img_path,"is not exist!")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if self.transform:
                image = self.transform(image)  # 原生transform写法
                # image = self.transform(image=image)['image']
            imgs.append(image)

        #return class_id, torch.stack(imgs), path
        return  torch.stack(imgs),np.ones(4,dtype=np.int64)*class_id
        # return  torch.stack(imgs),class_id
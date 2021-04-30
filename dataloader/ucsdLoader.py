import numpy as np
import os
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import random

import cv2

from torchvision import transforms

ROOT_PATH = "../../ProcessedData/vidf-cvpr_UCSD"


class ucsdDataset(data.Dataset):
    def __init__(self, setname, args):
        if setname == 'train':
            self.img_path = "../ProcessedData/vidf-cvpr_UCSD/train/images"
            self.gt_path = "../ProcessedData/vidf-cvpr_UCSD/train/GT"

        else:
            self.img_path = "../ProcessedData/vidf-cvpr_UCSD/test/images"
            self.gt_path = "../ProcessedData/vidf-cvpr_UCSD/test/GT"

        self.phase = setname
        self.gt_downsample = 16
        self.LOG_PARA = args.LOG_PARA

        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]

        self.num_samples = len(self.data_files)
        self.seg_len = args.seg_len


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        if index < (len(self) - (self.seg_len - 1)):
            start = index
            end = index + self.seg_len

            indices_item = list(range(start, end))
            imgs = []
            dens = []

            fname = self.data_files[0]
            img, den = self.read_image_and_gt(fname)

            ds_rows=int(img.shape[0]//(self.gt_downsample))
            ds_cols=int(img.shape[1]//(self.gt_downsample)) 

            crop_size = (ds_cols * self.gt_downsample//2, ds_rows * self.gt_downsample//2)

            dx = int(random.random()*ds_cols*self.gt_downsample*1./2)
            dy = int(random.random()*ds_rows*self.gt_downsample*1./2)
            rand_num = random.randint(0, 1) 
            for i in indices_item:

                fname = self.data_files[i]
                img, den = self.read_image_and_gt(fname) 
                
                ds_rows=int(img.shape[0]//(self.gt_downsample))
                ds_cols=int(img.shape[1]//(self.gt_downsample))
                img = cv2.resize(img, (ds_cols*self.gt_downsample, ds_rows*self.gt_downsample))
                den = cv2.resize(den, (ds_cols*self.gt_downsample, ds_rows*self.gt_downsample))

                if self.phase == 'train':
                    img = img[dy:crop_size[1]+dy, dx:crop_size[0]+dx]
                    den = den[dy:crop_size[1]+dy, dx:crop_size[0]+dx]
                    if rand_num == 1:
                        img = img[:,::-1].copy()
                        den = den[:,::-1].copy()

                img = img.transpose((2,0,1))

                img_tensor = torch.tensor(img, dtype=torch.float)
                img_tensor = transforms.functional.normalize(img_tensor,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

                den_tensor = torch.tensor(den, dtype=torch.float)
                den_tensor = den_tensor * self.LOG_PARA

                imgs.append(img_tensor)
                dens.append(den_tensor)

            return imgs[0], dens[0]

        else:
            fname = self.data_files[index]
            img, den = self.read_image_and_gt(fname)   

            ds_rows=int(img.shape[0]//(self.gt_downsample))
            ds_cols=int(img.shape[1]//(self.gt_downsample))
            img = cv2.resize(img,(ds_cols*self.gt_downsample, ds_rows*self.gt_downsample))             
            den = cv2.resize(den,(ds_cols*self.gt_downsample, ds_rows*self.gt_downsample))  
                   
       
            crop_size = (img.shape[1]//2, img.shape[0]//2)
            dx = int(random.random()*img.shape[1]*1./2)
            dy = int(random.random()*img.shape[0]*1./2)
            rand_num = random.randint(0, 1)

            if self.phase == 'train':
                img = img[dy:crop_size[1]+dy, dx:crop_size[0]+dx]
                den = den[dy:crop_size[1]+dy, dx:crop_size[0]+dx]
                if rand_num == 1:
                    img = img[:,::-1].copy()
                    den = den[:,::-1].copy()


            img = img.transpose((2,0,1))

            img_tensor = torch.tensor(img, dtype=torch.float)
            img_tensor = transforms.functional.normalize(img_tensor,mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])

            den_tensor = torch.tensor(den, dtype=torch.float)
            den_tensor = den_tensor * self.LOG_PARA

            return img_tensor, den_tensor
    
    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self, fname):
        img = plt.imread(os.path.join(self.img_path, fname))/255

        if len(img.shape)==2:
            img = img[:,:,np.newaxis]
            img = np.concatenate((img, img, img), 2)
        den = np.load(os.path.join(self.gt_path, os.path.splitext(fname)[0] + '.npy'))

        return img, den



    def get_num_samples(self):
        return self.num_samples











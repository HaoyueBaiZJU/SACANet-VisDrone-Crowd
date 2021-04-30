import numpy as np
import os
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import random

import cv2

from torchvision import transforms

ROOT_PATH = "../ProcessedData/shanghaitech_part_B"


class shhbDataset(data.Dataset):
    def __init__(self, setname, args):

        if setname == 'train':
 
            self.img_path = "../ProcessedData/shanghaitech_part_B/train/images"
            self.gt_path = "../ProcessedData/shanghaitech_part_B/train/ground_truth"
        else:

            self.img_path = "../ProcessedData/shanghaitech_part_B/test/images"
            self.gt_path = "../ProcessedData/shanghaitech_part_B/test/ground_truth"
        self.phase = setname
        self.gt_downsample = 8
        self.LOG_PARA = args.LOG_PARA
        self.data_files = [filename for filename in os.listdir(self.img_path) \
                           if os.path.isfile(os.path.join(self.img_path,filename))]
        self.num_samples = len(self.data_files)


    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
 
        fname = self.data_files[index]
        img, den = self.read_image_and_gt(fname)


        if self.phase == 'train':

            crop_size = (224, 224)

            dx = int(random.random()*(img.shape[1] - 224))
            dy = int(random.random()*(img.shape[0] - 224))
         
            img = img[dy:crop_size[1]+dy, dx:crop_size[0]+dx]
            den = den[dy:crop_size[1]+dy, dx:crop_size[0]+dx]

            if random.randint(0,1) == 1:
                img = img[:,::-1]
                den = den[:,::-1]

        
        ds_rows=int(img.shape[0]//(self.gt_downsample))
        ds_cols=int(img.shape[1]//(self.gt_downsample))
        img = cv2.resize(img, (ds_cols*self.gt_downsample, ds_rows*self.gt_downsample))
             
        img = img.transpose((2,0,1))
        den = cv2.resize(den, (ds_cols, ds_rows))

        den = den[np.newaxis,:,:] * self.gt_downsample * self.gt_downsample


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











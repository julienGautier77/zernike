#! /home/zita1/miniforge3/bin/python3.10
import os 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from skimage import io
import numpy as np
import matplotlib as plt

class DataPreparation(Dataset):
    def __init__(self,root_dir):
        """
          Arguments:
                file (str) : Image name
                root_dir (str) : Directory
        """
        super().__init__()
        #self.file = file
        self.root = root_dir
    
    def __len__(self):
        return 7

    def __getitem__(self, idx):
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        print('idx',idx)
        if int(idx) < 10:
            num = "0000"+ str(idx)
        elif 9<idx<100:
            num = "000" + str(idx)
        elif 99<idx<1000:
            num = "00" + str(idx)
        elif 999<idx<10000:
            num = "0" + str(idx)
        else:
            num = "0" + str(idx)
        
        self.file = 'img'+num
        for f in os.listdir(self.root):
            if self.file == f.split("_")[0] :
                self.file = f

        print ('file name : ',self.file)

        img_name = os.path.join(self.root,self.file)
        
        zernike_coef = img_name[:-5]
        zernike_coef = zernike_coef.split("_")
        nb_img = zernike_coef [0].split("/")[1]
        nb_img = int(nb_img[3:])
        zernike_coef = (zernike_coef[1:])
        zernike_coef =np.asarray(zernike_coef,dtype=np.float32)
        img  = io.imread(img_name)
        img=img/img.max() # normalization 
        sample ={'image':img, 'labels' : zernike_coef}
        self.zernike_coef=zernike_coef
        
        return {'image':(img),'labels':(zernike_coef)}
        

data = DataPreparation(root_dir='data2')
dataLoader = DataLoader(data,batch_size=4,shuffle=True,num_workers=0)
x_train=[]
y_train=[]
for i in range (0,10000
                ):

    x_train.append([data[i]['image']])
    y_train.append([data[i]['labels']])
x_train = np.asarray(x_train)
#x_train = torch.from_numpy(x_train)
y_train = np.asarray(y_train)
#y_train = torch.from_numpy(y_train)
print(x_train.shape,x_train.min(),x_train.max())
print(y_train.shape,y_train.min(),y_train.max())

np.save('x_train2',x_train)
np.save('y_train2',y_train)

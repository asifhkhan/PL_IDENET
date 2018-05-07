#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  2 22:58:26 2018

@author: Stamatis Lefkimmiatis
@email : s.lefkimmatis@skoltech.ru
"""

import os
import numpy as np
import torch.utils.data as data
from pydl.utils import gen_imdb_BSDS500_fromList
from matplotlib import image as Img

class BSDS(data.Dataset):
    
    def __init__(self,stdn,random_seed=20180102,filepath='',train=True,\
                 color=False,shape=(180,180),im2Tensor = True):
        
        assert(isinstance(stdn,(float,int,tuple))),"stdn is expected to be either "\
        +"a float or an int or a tuple"
                
        if isinstance(stdn,float):
            stdn = (stdn,)
        if isinstance(stdn,tuple):
            stdn = tuple(float(i) for i in stdn)
        
        self.stdn = np.asarray(stdn) 
        self.train = train
        self.rng = np.random.RandomState(random_seed)
        
        if im2Tensor:
            fshape = (3,2,0,1)
        else:
            fshape = (3,0,1,2)
              
        if self.train:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.train_gt = f['train_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.train_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='train').transpose(fshape)
            
            self.train_data = self.generate_NoisyData()
        else:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.test_gt = f['test_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.test_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='test').transpose(fshape)
            
            self.test_data = self.generate_NoisyData()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, noise_std).
        """
        
        if self.train:
            img, target, noise_std = self.train_data[index],\
                                     self.train_gt[index%len(self.train_gt)],\
                                     self.stdn.astype(self.train_gt.dtype)[index//len(self.train_gt)]
        else:
            img, target, noise_std = self.test_data[index],\
                                     self.test_gt[index%len(self.test_gt)],\
                                     self.stdn.astype(self.test_gt.dtype)[index//len(self.test_gt)]
        
        return img,target,noise_std
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def generate_NoisyData(self):
        r"""Create noisy observations using the ground-truth data."""
        if self.train:
            input = self.train_gt
        else:
            input = self.test_gt
            
        shape = input.shape
        dtype = input.dtype
                        
        ndata_shape = (shape[0]*len(self.stdn),)+shape[1:]
        ndata = np.empty(ndata_shape,dtype=dtype)
        
        for i in range(len(self.stdn)):
            noise = self.stdn[i]*self.rng.randn(*shape)
            noise = noise.astype(dtype)
            ndata[shape[0]*i:shape[0]*(i+1),...] = input+noise
        
        return ndata

class BSDS_v2(data.Dataset):
    
    def __init__(self,stdn,random_seed=20180102,filepath='',train=True,\
                 color=False,shape=(180,180),im2Tensor = True):
        
        assert(isinstance(stdn,(float,int,tuple))),"stdn is expected to be either "\
        +"a float or an int or a tuple"
                
        if isinstance(stdn,(int,float)):
            stdn = (stdn,)
        if isinstance(stdn,tuple):
            stdn = tuple(float(i) for i in stdn)
        
        self.stdn = np.asarray(stdn) 
        self.train = train
        self.rng = np.random.RandomState(random_seed)
        
        if im2Tensor:
            fshape = (3,2,0,1)
        else:
            fshape = (3,0,1,2)
              
        if self.train:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.train_gt = f['train_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.train_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='train').transpose(fshape)
            
            self.train_data = self.generate_NoisyData()
            self.train_obs = self.train_data
        else:
            if os.path.isfile(filepath):
                f = np.load(filepath)
                self.test_gt = f['test_set'].transpose(fshape)
            else:
                currentPath = os.path.dirname(os.path.realpath(__file__))
                listPath = os.path.join(currentPath,"../../datasets/BSDS500/BSDS_validation_list.txt")
                imdbPath = os.path.join(currentPath,"../../datasets/BSDS500/")
                self.test_gt = gen_imdb_BSDS500_fromList(color=color,\
                                    listPath = listPath, imdbPath = imdbPath,\
                                    shape=shape,data ='test').transpose(fshape)
            
            self.test_data = self.generate_NoisyData()
            self.test_obs = self.test_data
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, noise_std).
        """
        
        if self.train:
            img, target, noise_std = self.train_data[index],\
                                     self.train_gt[index%len(self.train_gt)],\
                                     self.stdn.astype(self.train_gt.dtype)[index//len(self.train_gt)]
            
            #obs = np.nan if self.train_obs is None else self.train_obs[index]
            obs = self.train_obs[index]   
        else:
            img, target, noise_std = self.test_data[index],\
                                     self.test_gt[index%len(self.test_gt)],\
                                     self.stdn.astype(self.test_gt.dtype)[index//len(self.test_gt)]

            #obs = np.nan if self.test_obs is None else self.test_obs[index]
            obs = self.test_obs[index]
            
        return img,target,noise_std,obs
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def generate_NoisyData(self):
        r"""Create noisy observations using the ground-truth data."""
        if self.train:
            input = self.train_gt
        else:
            input = self.test_gt
            
        shape = input.shape
        dtype = input.dtype
                        
        ndata_shape = (shape[0]*len(self.stdn),)+shape[1:]
        ndata = np.empty(ndata_shape,dtype=dtype)
        
        for i in range(len(self.stdn)):
            noise = self.stdn[i]*self.rng.randn(*shape)
            noise = noise.astype(dtype)
            ndata[shape[0]*i:shape[0]*(i+1),...] = input+noise
        
        return ndata

class BSDS68(data.Dataset):
    
    def __init__(self,stdn,random_seed=20180102,tall=True,color=False,\
                 im2Tensor=True,dtype='f', filename=\
                 "../../datasets/BSDS500/BSDS_validation_list.txt"):
        
        assert(isinstance(stdn,(float,int,tuple))),"stdn is expected to be either "\
        +"a float or an int or a tuple"
                
        if isinstance(stdn,(float,int)):
            stdn = (stdn,)
        if isinstance(stdn,tuple):
            stdn = tuple(float(i) for i in stdn)
        
        self.stdn = np.asarray(stdn) 
        self.tall = tall
        self.rng = np.random.RandomState(random_seed)
        
        if im2Tensor:
            fshape = (3,2,0,1)
        else:
            fshape = (3,0,1,2)
               
        currentPath = os.path.dirname(os.path.realpath(__file__))
        filename = os.path.join(currentPath,filename)
        dbPath = os.path.dirname(filename)
        
        with open(filename) as f:
            imList = f.readlines()
            imList = [f.strip() for f in imList]

        if color:
            dbPath = os.path.join(dbPath,'color')
        else:
            dbPath = os.path.join(dbPath,'gray')
        
        
        img = np.ndarray(0)
        if tall:
            for i in imList:
                tmp = Img.imread(os.path.join(dbPath,i)).astype(dtype)
                tmp.shape += (1,)
                if tmp.shape[0] > tmp.shape[1]:
                    img = np.concatenate((img,tmp),axis=3) if img.size else tmp
        else:
            for i in imList:
                tmp = Img.imread(os.path.join(dbPath,i)).astype(dtype)
                tmp.shape += (1,)
                if tmp.shape[0] <= tmp.shape[1]:
                    img = np.concatenate((img,tmp),axis=3) if img.size else tmp
                    
        
        self.img_gt = img.transpose(fshape)
        self.img_noisy = self.generate_NoisyData()
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (img, target, noise_std).
        """
        
        img, target, noise_std = self.img_noisy[index],\
                                 self.img_gt[index%len(self.img_gt)],\
                                 self.stdn.astype(self.img_gt.dtype)[index//len(self.img_gt)]
        
        return img,target,noise_std
    
    def __len__(self):
        
        return len(self.img_noisy)
    
    def generate_NoisyData(self):
        r"""Create noisy observations using the ground-truth data."""
            
        shape = self.img_gt.shape
        dtype = self.img_gt.dtype
                        
        ndata_shape = (shape[0]*len(self.stdn),)+shape[1:]
        ndata = np.empty(ndata_shape,dtype=dtype)
        
        for i in range(len(self.stdn)):
            noise = self.stdn[i]*self.rng.randn(*shape)
            noise = noise.astype(dtype)
            ndata[shape[0]*i:shape[0]*(i+1),...] = self.img_gt+noise
        
        return ndata
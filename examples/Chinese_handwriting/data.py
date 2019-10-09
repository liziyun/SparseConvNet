
# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchnet
import sparseconvnet as scn
import pickle
import math
import random
import numpy as np
import os

if not os.path.exists('pickle/'):
    #print('Downloading and preprocessing data ...')
    #os.system(
    #    'wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1trn_pot.zip')
    #os.system(
    #    'wget http://www.nlpr.ia.ac.cn/databases/download/feature_data/OLHWDB1.1tst_pot.zip')
    os.system('mkdir -p POT/ pickle/')
    os.system('unzip OLHWDB1.1trn_pot.zip -d POT/')
    os.system('unzip OLHWDB1.1tst_pot.zip -d POT/')
    os.system('python readPotFiles.py')

def interp(sample,x,y):
    return torch.from_numpy(np.hstack([np.interp(sample.numpy(),x.numpy(),y[:,i].numpy())[:,None] for i in range(y.shape[1])])).float()
class Data(torch.utils.data.Dataset):
    def __init__(self,file,scale=63):
        print('Loading', file, 'and balancing points for scale', scale)
        torch.utils.data.Dataset.__init__(self)
        self.data = pickle.load(open(file, 'rb'))
        for j in range(len(self.data)):
            strokes=[]
            features=[]
            for k,stroke in enumerate(self.data[j]['input']):
                #print("original\n")
                #print(stroke.shape)
                #print(stroke)
                if len(stroke)>1:
                    stroke=stroke.float()/255-0.5
                    stroke*=scale-1e-3
                    #print("scaled\n")
                    #print(stroke)
                    delta=stroke[1:]-stroke[:-1]
                    #print("delta\n")
                    #print(delta)
                    mag=(delta**2).sum(1)**0.5
                    #print("mag\n")
                    #print(mag)
                    l=mag.cumsum(0)
                    #print("l\n")
                    #print(l)
                    zl=torch.cat([torch.zeros(1),l])
                    #print("zl\n")
                    #print(zl)
                    #print("arange\n")
                    #print(torch.arange(0,zl[-1]))
                    strokes.append(interp(torch.arange(0,zl[-1]),zl,stroke))
                    #print("strokes new\n")
                    #print(strokes)
                    delta/=mag[:,None]
                    delta=torch.Tensor(delta[[i//2 for i in range(2*len(l))]])
                    zl_=zl[[i//2 for i in range(1,2*len(l)+1)]]
                    features.append(interp(torch.arange(0,zl[-1]),zl_,delta))
                    #print("coords\n")
                    #print(strokes)
                    #print("features\n")
                    #print(features)
            self.data[j]['coords'] = torch.cat(strokes,0)
            self.data[j]['features'] = torch.cat(features,0)
        for i, x in enumerate(self.data):
            x['idx'] = i
        print('Loaded', len(self.data), 'points')
    def __getitem__(self,n):
        return self.data[n]
    def __len__(self):
        return len(self.data)

def MergeFn(spatial_size=63):
    center = spatial_size/2
    def merge(tbl):
        v=torch.Tensor([[1,0,0]])
        targets=[x['target'] for x in tbl]
        locations=[]
        features=[]
        for idx,char in enumerate(tbl):
            coords=char['coords']+center
            coords = torch.cat([coords.long(),torch.LongTensor([idx]).expand([coords.size(0),1])],1)
            locations.append(coords)
            print("locations\n")
            print(locations)
            f=char['features']
            f = torch.cat([f,torch.ones([f.size(0),1])],1)
            features.append(f)
            print("features\n")
            print(features)
        return {'input': scn.InputLayerInput(torch.cat(locations,0), torch.cat(features,0)), 'target': torch.LongTensor(targets)}
    return merge


def get_iterators(*args):
    #return {'train': torch.utils.data.DataLoader(Data('pickle/train.pickle'), collate_fn=MergeFn(), batch_size=100, shuffle=True, num_workers=10),
    #        'val': torch.utils.data.DataLoader(Data('pickle/test.pickle'), collate_fn=MergeFn(), batch_size=100, shuffle=True, num_workers=10)}
    return {'train': torch.utils.data.DataLoader(Data('pickle/train.pickle'), collate_fn=MergeFn(), batch_size=32, shuffle=True, num_workers=10),
            'val': torch.utils.data.DataLoader(Data('pickle/test.pickle'), collate_fn=MergeFn(), batch_size=32, shuffle=True, num_workers=10)}

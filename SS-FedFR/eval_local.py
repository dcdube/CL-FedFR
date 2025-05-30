# coding: utf-8
import os
import pickle
import pandas as pd
# import matplotlib.pyplot as plt
import sklearn
import argparse
from sklearn.metrics import roc_curve, auc
from prettytable import PrettyTable
from pathlib import Path
import sys
import warnings
from tqdm import tqdm
import torch.nn as nn
import mxnet as mx
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from collections import defaultdict
import random
import cv2
import time
from math import ceil
import copy

def set_random_seed(seed_value, use_cuda=True,deterministic=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    # torch.use_deterministic_algorithms(True)
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        if deterministic == True:
            torch.backends.cudnn.deterministic = True  #needed
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

class MXFaceDataset(Dataset):
    def __init__(self, root_dir, transform):
        super(MXFaceDataset, self).__init__()
        path_imgrec = os.path.join(root_dir, 'test.rec')
        path_imgidx = os.path.join(root_dir, 'test.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        # process header
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
            self.num_classes = self.header0[1]
        else:
            self.imgidx = np.array(list(self.imgrec.keys))
            self.num_classes = None
        # transform
        self.transform = transform

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.imgidx)


class CallBack_LocalVerifi(object):
    def __init__(self,frequent,rank,data_dir,th=-1,flip_test=False,output_dir=None,verbose=True,workers=2,batch_size=800):
        self.frequent = frequent
        self.rank = rank
        self.data_dir = data_dir
        self.th = th
        self.flip_test = flip_test
        self.output_dir = output_dir
        self.client_record = defaultdict(list)
        self.verbose = verbose
        self.workers = workers
        self.batch_size = batch_size

    def veri_test(self, backbone_orign, global_step, ID_list, client_ID):
        if self.rank is 0 and global_step >= self.th and global_step % self.frequent == 0:
            client_dir = os.path.join(self.output_dir,'clients','client_%d'%(client_ID))
            os.system('mkdir -p %s'%(client_dir))
            backbone = copy.deepcopy(backbone_orign)
            backbone.eval()
            img_feats,labels = self.generate_features(backbone)
            feat_path = os.path.join(client_dir,'img_feats.npy')
            label_path = os.path.join(client_dir,'labels.npy')
            np.save(feat_path,img_feats)
            np.save(label_path,labels)
            gpu_counts = min(2,torch.cuda.device_count())
            os.system('python3 roc_cuda.py --feat_path %s --label_path %s --output_dir %s --ID_s_e %d %d --epoch %d --workers %d'\
                %(feat_path,label_path,client_dir,ID_list[0],ID_list[-1]+1,global_step,gpu_counts))
            os.remove(feat_path)
            os.remove(label_path)
            backbone.cpu()
            del backbone
        return

    @torch.no_grad()
    def generate_features(self,backbone): 
        ## prepare dataset
        batch_size = 512
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        test_dataset = MXFaceDataset(self.data_dir, test_transform)
        test_dataloader = DataLoader(test_dataset,batch_size=batch_size,shuffle=False,num_workers=4,drop_last=False)

        ## create meta
        idx_id_pair = pd.read_csv(os.path.join(self.data_dir,'idx_id_pair.txt'), sep=' ',dtype=np.int32).values
        idx_id_pair[:,0] -= 1
        labels = idx_id_pair[:,1].flatten()

        if self.verbose:
            print('load idx_id_pair', idx_id_pair.shape)
            print('Generating image features...')
        tic = time.time() 
        backbone = backbone.to(self.rank)
        backbone = torch.nn.DataParallel(backbone)
        backbone.eval()
        if self.flip_test == False:
            img_feats = np.empty((len(test_dataset), 512), dtype=np.float32)
        else:
            img_feats = np.empty((len(test_dataset), 1024), dtype=np.float32)
        for i,batch_img in enumerate(test_dataloader):
            if (i+1)%100 == 0 and self.verbose:
                print('%d/%d'%(i+1,len(test_dataloader)))
            if self.flip_test == False:
                output_feats = backbone(batch_img.to(self.rank)).cpu().numpy()
                img_feats[i*batch_size: (i+1)*batch_size] = output_feats
            else:
                output_feats = backbone(batch_img.to(self.rank)).cpu().numpy()
                output_feats_flip = backbone(torch.fliplr(batch_img).to(self.rank)).cpu().numpy()
                img_feats[i*batch_size:(i+1)*batch_size] = np.concatenate([output_feats,output_feats_flip],axis=1)
        if self.flip_test:
            img_feats = img_feats[:,:img_feats.shape[1]//2] + img_feats[:,img_feats.shape[1]//2:]
        img_feats = sklearn.preprocessing.normalize(img_feats)
        toc =time.time()
        backbone = backbone.module.cpu()
        torch.cuda.empty_cache()
        if self.verbose:
            print('Takes %.2f sec to generate image features'%(toc-tic),img_feats.shape)
        return img_feats,labels

if __name__ == '__main__':

    set_random_seed(100)
    import backbones
    import argparse
    parser = argparse.ArgumentParser('')
    parser.add_argument('--backbone',type=str,default='multi')
    parser.add_argument('--path')
    parser.add_argument('--epoch',type=int)
    parser.add_argument('--num_client',type=int)
    args = parser.parse_args()


    backbone = eval("backbones.{}".format('sphnet'))(False, dropout=0, fp16=False)
    # backbone.eval()
    # callback = CallBack_LocalVerifi(1, 0, '/master/SS-FedFR/ms1m_split/local_veri_start_4000',output_dir='./ckpt/test/',flip_test=False)
    # callback.veri_test(backbone,0,list(range(0,100)),0)

    

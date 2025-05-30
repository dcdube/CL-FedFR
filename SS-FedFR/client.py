import torch
from config import config as cfg
import backbones
import logging
import losses
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data.distributed
from utils.utils_logging import AverageMeter, init_logging
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from eval_local import CallBack_LocalVerifi
from tqdm import tqdm
import gc
import copy
import pickle
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import MXFaceDataset_Subset,MXFaceDataset_Combine,DataLoaderX
from functools import reduce
from backbones import BottleBlock
from dataset import MXFaceDataset_Split_PseudoLabeled
class BCE_module(nn.Module):
    def __init__(self,hidden,n_class,converter_layer=1,m=0.4,r=30.0,t=3):
        super(BCE_module,self).__init__()
        converter = []
        if converter_layer == 1:
            layer = nn.Linear(hidden,hidden)
            nn.init.eye_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
            converter.append(layer)
            self.converter = nn.Sequential(*converter)
        else:
            self.converter = BottleBlock(hidden, 4)

        self.weight = nn.Parameter(torch.normal(0,0.01,(n_class,hidden)))
        self.bias = nn.Parameter(torch.zeros(n_class))
        self.g_func = lambda x : (2*(((x+1)/2).pow(t))-1)
        self.n_class = n_class
        self.hidden = hidden
        self.m = m
        self.r = r
    def forward(self,x,labels):
        feat = self.converter(x)
        cosine = torch.matmul(F.normalize(feat), F.normalize(self.weight).t())
        gt = torch.zeros(len(x),self.n_class+1,device=x.device).bool()
        tmp_labels = labels.clone()
        tmp_labels[tmp_labels >= self.n_class] = self.n_class
        gt[torch.arange(len(x)),tmp_labels] = True
        gt = gt[:,:-1]
        # positive
        cosine[gt] = self.r * (self.g_func(cosine[gt])-self.m)
        # negative
        cosine[~gt] = self.r * (self.g_func(cosine[~gt])+self.m)
        cosine += self.bias.unsqueeze(0)
        return cosine, gt
    def initialize(self,fc):
        self.weight.data = fc.clone()


class FC_module(nn.Module):
    def __init__(self,hidden,n_class,output_dir):
        super(FC_module,self).__init__()
        self.fc = nn.Parameter(torch.normal(0,0.01,(n_class,hidden)))
        self.output_dir = output_dir
        self.n_class = n_class
    def forward(self,x,normalize_feat=True):
        if normalize_feat:
            output = torch.matmul(F.normalize(x),F.normalize(self.fc).t())
        else:
            output = torch.matmul(x,F.normalize(self.fc).t())
        return output

    def update_from_tensor(self,fc):
        self.fc.data = fc.clone()
    def update_with_pretrain(self,pretrain_fc):
        self.fc = nn.Parameter(torch.cat([self.fc.data,pretrain_fc],dim=0))
    def remove_pretrain(self):
        self.fc.data = self.fc.data[0:self.n_class]
    def get_pretrain_fc(self):
        return self.fc.data[self.n_class:]

class Branch_model(nn.Module):
    def __init__(self,backbone,fc_module,bce_module):
        super(Branch_model,self).__init__()
        self.backbone = backbone
        self.fc_module = fc_module
        self.bce_module = bce_module
    def forward(self,imgs,labels,contrastive=False,detach=False):
        feature = self.backbone(imgs)
        cosface_logits = self.fc_module(feature)
        if detach:
            bce_logits,bce_gts = self.bce_module(feature.detach(),labels)
        else:    
            bce_logits,bce_gts = self.bce_module(feature,labels)
        if contrastive:
            return cosface_logits, bce_logits, bce_gts,feature
        return cosface_logits, bce_logits, bce_gts

class Sequential_model(nn.Module):
    def __init__(self,backbone,fc_module):
        super(Sequential_model,self).__init__()
        self.backbone = backbone
        self.fc_module = fc_module
    def forward(self,imgs,contrastive=False):
        feature = self.backbone(imgs)
        logits = self.fc_module(feature)
        if contrastive:
            return logits,feature
        else:
            return logits


class Client(object):
    def __init__(self, cid, args, data):
        self.cid = cid
        self.args = args
        self.num_classes = data.train_class_sizes[self.cid] 
        self.local_epoch = args.local_epoch
        self.dataset_size = data.train_dataset_sizes[self.cid]
        self.train_loader = data.train_loaders[self.cid]
        self.train_unlabeled_loader = data.train_unlabeled_loaders[self.cid]
        # The global base ID for each client (ex. local: 0-99, global: 300-399 )
        self.ID_base = data.train_loaders[self.cid].dataset.ID_base
        self.target_ID = list(range(self.ID_base,self.ID_base+self.num_classes))

        if hasattr(data, 'test_loaders'):
            self.test_loaders = data.test_loaders[self.cid]
        if hasattr(data, 'public_train_loader'):
            self.public_num_classes = data.public_train_loader.dataset.num_classes
        # add margin
        self.margin_softmax = eval("losses.{}".format(args.loss))(s=30,m=0.4)
        if self.args.BCE_local:
            self.bce_module = BCE_module(512, self.num_classes,cfg.converter_layer)
            self.bce_loss = losses.BCE_loss()

        ### distributed
        self.rank = 0 #dist.get_rank()
        self.local_rank = 0 #args.local_rank
        
        self.dropout = 0.4 if cfg.dataset is "webface" else 0
        self.backbone_state_dict = None
        
        ### Create directory 
        self.client_output = os.path.join(args.output_dir,'clients','client_%d'%(self.cid))

        ### FC module, on cpu 
        self.fc_module = FC_module(512,self.num_classes,self.client_output)

        ### contrastive backbone (bb)
        if self.args.contrastive_bb:
            self.last_model = eval("backbones.{}".format(self.args.network))(False, dropout=self.dropout, fp16=cfg.fp16)
            self.con_criterion = nn.CosineSimilarity(dim=1)
            self.temperature = 0.5

        self.logger = logging.getLogger('FL_face.client')
        self.semilogger = logging.getLogger('semilogger.cid{}'.format(cid))
        self.semilogger.setLevel(logging.INFO)
        log_dir = '/home/master/SS-FedFR/ckpt/FedFR/clients/client_' + str(self.cid)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_file_1 = os.path.join(log_dir, 'semi_supervised_stats.log')
        file_handler1 = logging.FileHandler(log_file_1,mode="w")
        file_handler1.setLevel(logging.DEBUG)
        formatter1 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler1.setFormatter(formatter1)
        self.semilogger.addHandler(file_handler1)
        self.semilogger.info("# of unlabeled samples: %d" % (data.train_unlabeled_dataset_sizes[self.cid]))
        self.semilogger.info("# of labeled samples: %d" % (data.train_dataset_sizes[self.cid]))
        self.counter = 0

    def data_update_fc(self,fed_model_state_dict,norm_before_avg,fc_name='center_features',save_to_disk=False):
        with torch.no_grad():
            ### Creating model
            backbone = eval("backbones.{}".format(self.args.network))(False, dropout=self.dropout, fp16=cfg.fp16)
            backbone.load_state_dict(fed_model_state_dict)
            backbone = nn.DataParallel(backbone)
            backbone.to(self.local_rank)
            backbone.eval()

            ### Local forward data
            init_fc = torch.zeros_like(self.fc_module.fc.data).to(self.local_rank)
            num_samples = torch.zeros(self.num_classes).to(self.local_rank)

            for step, (img, label) in enumerate(self.test_loaders):
                features = backbone(img)
                if norm_before_avg:
                    features = F.normalize(features)
                u_label = torch.unique(label)
                for l in u_label:
                    init_fc[l:l+1,:] += torch.sum(features[label==l,:],dim=0)
                    num_samples[l] += torch.sum(label==l)
            ### average features
            init_fc /= num_samples.unsqueeze(1)
            init_fc = init_fc.cpu()
            
            if save_to_disk:
                torch.save(init_fc,os.path.join(self.client_output,fc_name+'.pth'))

            self.fc_module.update_from_tensor(init_fc)
            del backbone,num_samples

    ## feature-based
    def choose_hard_negative_2(self,public_train_loader,pretrained_label,pretrained_feats,threshold=0.2):
        public_loader_subset = copy.deepcopy(public_train_loader)
        
        ## forward local data
        local_feats = []
        backbone = eval("backbones.{}".format(self.args.network))(False, dropout=self.dropout, fp16=cfg.fp16)
        backbone.load_state_dict(self.backbone_state_dict)
        backbone.eval()
        backbone.to(self.local_rank)
        #
        with torch.no_grad():
            for step, (img, label) in enumerate(self.test_loaders):
                features = F.normalize(backbone(img.to(self.local_rank)))
                local_feats.append(features.cpu())
            local_feats = torch.cat(local_feats,dim=0)
        backbone = backbone.cpu()
        
        similarity = torch.matmul(local_feats,pretrained_feats.t())
        unique_idx = []
        times = 100  ## prevent out of mem of RAM (when torch.where())
        batch = len(similarity)//times +1 
        for i in range(times):
            unique_idx.append(torch.where(similarity[i*batch:(i+1)*batch]>threshold)[1].numpy())
        ## do Union
        unique_idx = sorted(reduce(np.union1d,unique_idx))
        ## update "imgidx" in dataset
        public_loader_subset.dataset.imgidx = np.array(unique_idx)+1
        num_ID = len(torch.unique(pretrained_label[unique_idx]))
        self.logger.info('%d imgs (%d ID) are hard negative with similarity > %.2f'%(len(unique_idx),num_ID,threshold))

        del backbone,similarity
        gc.collect()
        torch.cuda.empty_cache()
        return public_loader_subset
    
    ## FC-based
    def choose_hard_negative(self,pretrain_fc,public_train_loader,pretrain_label,self_fc,threshold=0.2,batch_size=128):

        public_loader_subset = copy.deepcopy(public_train_loader)
        similarity = torch.matmul(F.normalize(self_fc),F.normalize(pretrain_fc).t())
        # n_class x 6000
        
        if isinstance(threshold, float):
            IDs = torch.unique(torch.where(similarity>threshold)[1]).numpy()
        if isinstance(threshold, int):
            # IDs = torch.argsort(simlarity,descending=True)[:threshold].numpy()
            raise NotImplementedError
        self.logger.info('%d ID are hard negative'%(len(IDs)))
        self.HN_ID = IDs

        ## return all FC
        pretrain_fc_subset = pretrain_fc
        # pretrain_fc_subset = pretrain_fc[IDs]

        # relabel_dict = dict()
        # for i in range(len(IDs)):
            # relabel_dict[IDs[i]] = i
        
        left_img_idx = []
        pretrain_label = pretrain_label.numpy()
        for i in range(len(pretrain_label)):
            if pretrain_label[i] in IDs:
                # imgidx start from 1
                left_img_idx.append(i+1)
        # alter dataset
        public_loader_subset.dataset.imgidx = np.array(left_img_idx)

        # class order is the same.
        # imgrec = public_loader_subset.dataset.imgrec
        # imgidx = np.array(left_img_idx)
        # num_classes = len(IDs)
        # transform = public_loader_subset.dataset.transform
        # tmp_dataset = MXFaceDataset_Subset(imgrec, imgidx, num_classes, relabel_dict, transform) 
        # public_loader_subset = DataLoader(tmp_dataset,batch_size=batch_size,shuffle=True,num_workers=2,\
                                            # pin_memory=True,drop_last=True)

        return pretrain_fc_subset,public_loader_subset

    def reweight_cosface(self,logits,labels):
        # logits : (B, C)
        # labels : B
        with torch.no_grad():
            idx_bool = torch.ones(logits.shape).bool()
            idx_bool[torch.arange(len(labels)),labels] = False
            tmp = logits.detach().clone()[idx_bool].reshape(len(labels),logits.shape[1]-1)[:,:self.num_classes].repeat(1,self.args.num_client-1)
            logits = torch.cat([logits,tmp],dim=1)

        # with torch.no_grad():
        #     idx_bool = torch.zeros(logits.shape).bool()
        #     for i in range(len(labels)):
        #         idx = torch.randperm(4000-self.num_classes)
        #         idx_bool[i,idx] = True
        #     balance = logits.detach().clone()[idx_bool].reshape(len(labels),4000-self.num_classes)
        # logits = torch.cat([logits,balance],dim=1)
        return logits

    def train_with_public_data(self,start_epoch=0,callback_verification=None,\
                                public_train_loader=None,pretrained_fc=None,choose_hard_negative=False,\
                                pretrained_label=None,pretrained_feats=None,threshold=1.01):
        # split ratio threshold
        ### Create hard negative dataloader
        if choose_hard_negative:
            public_loader_subset = self.choose_hard_negative_2(public_train_loader, pretrained_label, pretrained_feats,\
                                                               threshold=cfg.HN_threshold)
        else:
            public_loader_subset = public_train_loader

        ### Create backbone, load weight, put GPU
        backbone = eval("backbones.{}".format(self.args.network))(False, dropout=self.dropout, fp16=cfg.fp16)
        backbone.load_state_dict(self.backbone_state_dict)
        backbone.eval()
        backbone.to(self.local_rank)
        ### Update self FC module and put FC to gpu
        self.fc_module.eval()
        self.fc_module.to(self.local_rank)
        self.semilogger.info("---------------------------Round %d---------------------------------" % self.counter)
        self.counter += 1

        backbone.train()
        self.fc_module.train()

        self.fc_module.to("cpu")
        self.fc_module.update_with_pretrain(pretrained_fc)
        self.fc_module.to(self.local_rank)


        
        if self.args.BCE_local:
            ### Create BCE model
            self.bce_module.train()
            self.bce_module.to(self.local_rank)
            model = Branch_model(backbone, self.fc_module, self.bce_module)
        else:
            model = Sequential_model(backbone, self.fc_module)
        
        ### Contrastive backbone, to parallel
        if self.args.contrastive_bb:
            with torch.no_grad():
                global_model = nn.DataParallel(copy.deepcopy(backbone).to(self.local_rank)).eval()
                self.last_model = nn.DataParallel(self.last_model.to(self.local_rank)).eval()

        ### first local test 
        if callback_verification is not None and start_epoch == 0:
            self.logger.info('Pretrain Local testing')
            callback_verification.veri_test(backbone, -1, self.target_ID, self.cid)
        
        opt = torch.optim.SGD(params=model.parameters(),lr=cfg.lr,momentum=0.9,weight_decay=cfg.weight_decay)

        ### For different lr of backbone & BCE
        # if self.args.BCE_local:
        #     opt = torch.optim.SGD([{'params':model.backbone.parameters()},{'params':model.fc_module.parameters()}],
        #             lr=cfg.lr,momentum=0.9,weight_decay=cfg.weight_decay)
        #     opt_bce = torch.optim.SGD(params=model.bce_module.parameters(),lr=cfg.lr_func(start_epoch)*10*cfg.lr,\
        #                               momentum=0.9,weight_decay=cfg.weight_decay)
        # else:
        #     raise NotImplementedError()

        model = torch.nn.DataParallel(model)

        schler = torch.optim.lr_scheduler.StepLR(opt,cfg.train_decay,gamma=0.1)
        loss_meter = AverageMeter()
        cos_meter = AverageMeter()
        con_meter = AverageMeter()
        bce_meter = AverageMeter()
        
        ### Start train w/ combine 
        for epoch in range(start_epoch, start_epoch+self.local_epoch):
            self.semilogger.info("\n*****Epoch Start %d*****\n" % epoch)
            pseudo_labeled_dataset = self.get_pseudo_labeled_data(backbone, threshold)
            ### combine dataloader
            if self.args.combine_dataset:
                combine_dataset = MXFaceDataset_Combine(self.train_loader.dataset, pseudo_labeled_dataset,
                                                        public_loader_subset.dataset)
                combine_loader = DataLoader(combine_dataset, batch_size=cfg.com_batch_size, shuffle=True, num_workers=6,
                                            pin_memory=True, drop_last=True)
                ### Update dataset size, for FedAvg
                self.dataset_size = len(combine_dataset)
            else:
                raise NotImplementedError()
            self.logger.info('Epoch %d,Total Epoch %d, Total step : %d, lr=%.4f'%(epoch,start_epoch+self.local_epoch,\
                            len(combine_loader),schler.get_last_lr()[0]))
            pbar = tqdm(total=len(combine_loader),ncols=120,leave=True)
            if self.args.BCE_local:  ### train with BCE loss
                for step, (imgs, labels) in enumerate(combine_loader):
                    opt.zero_grad()
                    # opt_bce.zero_grad()
                    imgs = imgs.to(self.local_rank)
                    labels = labels.to(self.local_rank)
                    ### train w/ contrastive
                    if self.args.contrastive_bb:
                        with torch.no_grad():
                            global_feats = global_model(imgs)
                            last_feats = self.last_model(imgs)
                        cos_logits, bce_logits ,bce_gts , feats = model(imgs,labels,contrastive=True,detach=self.args.BCE_detach)
                        # Contrastive
                        pos_sim = self.con_criterion(feats,global_feats)/self.temperature
                        neg_sim = self.con_criterion(feats,last_feats)/self.temperature
                        con_label = torch.zeros(len(labels),device=pos_sim.device).long()
                        con_loss = F.cross_entropy(torch.stack([pos_sim,neg_sim],dim=1), con_label)
                        # Cosface
                        cos_logits = self.margin_softmax(cos_logits,labels)
                        if self.args.reweight_cosface:
                            cos_logits = self.reweight_cosface(cos_logits, labels)
                        cos_loss = F.cross_entropy(cos_logits, labels)
                        # bce loss
                        bce_loss = self.bce_loss(bce_logits,bce_gts)
                        loss = cos_loss + 10 * bce_loss + cfg.mu * con_loss
                        con_meter.update(con_loss.item(),1)
                    else:
                        cos_logits, bce_logits ,bce_gts = model(imgs,labels,contrastive=False,detach=self.args.BCE_detach)
                        # Cosface
                        cos_logits = self.margin_softmax(cos_logits,labels)
                        if self.args.reweight_cosface:
                            cos_logits = self.reweight_cosface(cos_logits, labels)
                        cos_loss = F.cross_entropy(cos_logits, labels)
                        # bce loss
                        bce_loss = self.bce_loss(bce_logits,bce_gts)
                        loss = cos_loss + 10 * bce_loss
                    loss.backward()
                    opt.step()
                    # opt_bce.step()
                    loss_meter.update(loss.item(),1)
                    cos_meter.update(cos_loss.item(),1)
                    bce_meter.update(bce_loss.item(),1)
                    if step > 10 and step % 40 == 0:
                        pbar.set_postfix(loss='%.3f,%.3f,%.3f,%.3f'%(loss_meter.avg,cos_meter.avg,con_meter.avg,bce_meter.avg))
                        self.logger.debug('Step %d, Loss : %.3f,%.3f,%.3f,%.3f'%(step,loss_meter.avg,cos_meter.avg,con_meter.avg,bce_meter.avg))
                    pbar.update(1)
            else:
                for step, (imgs, labels) in enumerate(combine_loader):
                    opt.zero_grad()
                    imgs = imgs.to(self.local_rank)
                    labels = labels.to(self.local_rank)
                    if self.args.contrastive_bb:
                        with torch.no_grad():
                            global_feats = global_model(imgs)
                            last_feats = self.last_model(imgs)
                        logits, feats = model(imgs)
                        # Contrastive
                        pos_sim = self.con_criterion(feats,global_feats)/self.temperature
                        neg_sim = self.con_criterion(feats,last_feats)/self.temperature
                        con_label = torch.zeros(len(labels),device=pos_sim.device).long()
                        con_loss = F.cross_entropy(torch.stack([pos_sim,neg_sim],dim=1), con_label)
                        #Cosface
                        logits = self.margin_softmax(logits,labels)
                        if self.args.reweight_cosface:
                            logits = self.reweight_cosface(logits, labels)
                        cos_loss = F.cross_entropy(logits, labels)
                        loss = cos_loss + cfg.mu*con_loss
                        con_meter.update(con_loss.item(),1)
                        cos_meter.update(cos_loss.item(),1)
                    else:                        
                        logits = model(imgs)
                        logits = self.margin_softmax(logits,labels)
                        if self.args.reweight_cosface:
                            logits = self.reweight_cosface(logits, labels)
                        loss = F.cross_entropy(logits, labels)

                    loss.backward()
                    opt.step()
                    loss_meter.update(loss.item(),1)
                    if step > 10 and step % 40 == 0:
                        pbar.set_postfix(loss='%.3f,%.3f,%.3f'%(loss_meter.avg,cos_meter.avg,con_meter.avg))
                        self.logger.debug('Step %d, Loss : %.3f,%.3f,%.3f'%(step,loss_meter.avg,cos_meter.avg,con_meter.avg))
                    pbar.update(1)
            pbar.close()
            schler.step()
            self.semilogger.info("*****Epoch End %d*****\n" % epoch)
        if self.rank is 0: self.logger.info("Client %d Ends: loss = %.3f"%(self.cid, loss_meter.avg))

        ### To CPU
        backbone = backbone.cpu()
        self.fc_module = self.fc_module.cpu()
        if self.args.BCE_local:
            self.bce_module = self.bce_module.cpu()
        if self.args.contrastive_bb:
            global_model = global_model.module.cpu()
            self.last_model = self.last_model.module.cpu()
        
        ### Local test
        if callback_verification is not None:
            self.logger.info("Client %d Local Testing"%(self.cid))
            os.system('mkdir -p %s'%(self.client_output))
            if self.args.BCE_local:
                backbone_converter = nn.Sequential(backbone,self.bce_module.converter)
                callback_verification.veri_test(backbone_converter, epoch, self.target_ID,self.cid)
                torch.save(backbone.state_dict(),os.path.join(self.client_output,'backbone.pth'))
                torch.save(self.bce_module.state_dict(),os.path.join(self.client_output,'bce_module.pth'))
            else:
                callback_verification.veri_test(backbone, epoch, self.target_ID,self.cid)
                torch.save(backbone.state_dict(),os.path.join(self.client_output,'backbone.pth'))
        else: ##others just save model
            os.system('mkdir -p %s'%(self.client_output))
            if self.args.BCE_local:
                torch.save(backbone.state_dict(),os.path.join(self.client_output,'backbone.pth'))
                torch.save(self.bce_module.state_dict(),os.path.join(self.client_output,'bce_module.pth'))
            else:
                torch.save(backbone.state_dict(),os.path.join(self.client_output,'backbone.pth'))

        
        self.backbone_state_dict = backbone.state_dict()
        
        if self.args.contrastive_bb:
            self.last_model.load_state_dict(backbone.state_dict())
            del global_model

        self.loss_meter = loss_meter
        # delete garbage
        del backbone
        gc.collect()
        torch.cuda.empty_cache()


    def train(self,start_epoch=0,callback_verification=None):
        # put model to gpu
        backbone = eval("backbones.{}".format(self.args.network))(False, dropout=self.dropout, fp16=cfg.fp16)
        backbone.load_state_dict(self.backbone_state_dict)
        backbone.to(self.local_rank)
        backbone.train()

        self.fc_module.to(self.local_rank)
        self.fc_module.train()

        model = nn.DataParallel(nn.Sequential(backbone,self.fc_module))
        # first test 
        if callback_verification is not None and start_epoch == 0:
            self.logger.info('Pretrain Local testing')
            callback_verification.veri_test(backbone, -1, self.target_ID, self.cid)
        
        opt = torch.optim.SGD(
            params=model.parameters(),
            lr=cfg.lr_func(start_epoch)*cfg.lr,momentum=0.9,weight_decay=cfg.weight_decay)

        loss_meter = AverageMeter()
        self.semilogger.info("---------------------------Round %d---------------------------------" % self.counter)
        self.counter += 1
        pseudo_labeled_dataset = self.get_pseudo_labeled_data(backbone, 0.0)

        for epoch in range(start_epoch, start_epoch+self.local_epoch):
            self.logger.info('Epoch %d, Total step : %d'%(epoch, len(self.train_loader)))
            pbar = tqdm(total=len(self.train_loader),ncols=120,leave=True)
            for step, (imgs, labels) in enumerate(self.train_loader):
                print(imgs.shape, " a ",labels.shape)
                opt.zero_grad()
                if len(imgs) ==1:
                    imgs = torch.cat([imgs,imgs],dim=0)
                    labels = torch.cat([labels,labels])
                imgs = imgs.to(self.local_rank)
                labels = labels.to(self.local_rank)
                logits = model(imgs)
                logits = self.margin_softmax(logits,labels)
                loss = F.cross_entropy(logits, labels)
                # pos cosine loss
                # output = F.relu(0.9 - self.fc_module(features))**2
                # loss = torch.mean(output)
                loss.backward()
                opt.step()
                loss_meter.update(loss.item(), 1)
                pbar.update(1)
                if step > 10 and step % 50 == 0:
                    pbar.set_postfix(loss='%.3f'%loss_meter.avg)
            pbar.close()

        if self.rank is 0: self.logger.info("Client %d Ends: loss = %.3f"%(self.cid, loss_meter.avg))
        backbone = backbone.cpu()
        self.fc_module = self.fc_module.cpu()
        self.loss_meter = loss_meter
        # callback
        if callback_verification is not None:
            self.logger.info("Client %d Local Testing"%(self.cid))
            os.system('mkdir -p %s'%(self.client_output))
            callback_verification.veri_test(backbone, epoch, self.target_ID,self.cid)
            torch.save(backbone.state_dict(),os.path.join(self.client_output,'backbone.pth'))

        self.backbone_state_dict = backbone.state_dict()
        # delete garbage
        del backbone
        gc.collect()

    def get_train_loss(self):
        return self.loss_meter.avg
    def get_model(self):
        return self.backbone_state_dict
    def get_global_fc(self):
        return self.fc_module.get_pretrain_fc()
    def get_model_path(self):
        return os.path.join(self.client_output,'backbone.pth')
    def get_data_size(self):
        return self.dataset_size

    def get_pseudo_labeled_data(self,backbone,threshold):
        images = torch.empty(0, 3, 112, 112, dtype=torch.float32).to(self.local_rank)
        pseudo_labels = torch.empty(0, dtype=torch.long).to(self.local_rank)
        labels = torch.empty(0, dtype=torch.long).to(self.local_rank)
        number_of_correct_labels = 0
        number_of_false_labels = 0
        number_of_correct_labels_b = 0
        number_of_false_labels_b = 0
        for step, (imgs, true_labels) in enumerate(self.train_unlabeled_loader):
            imgs = imgs.to(self.local_rank)
            true_labels = true_labels.to(self.local_rank)
            features = backbone(imgs)

            cos_logits = self.fc_module(features)
            max_logits, max_indices = torch.max(cos_logits, dim=1)
            for i in range(len(max_indices)):
                if true_labels[i] == max_indices[i]:
                    number_of_correct_labels_b += 1
                else:
                    number_of_false_labels_b += 1

            mask = max_logits > threshold
            imgs = imgs[mask]
            true_labels = true_labels[mask]

            max_indices = max_indices[mask]
            for i in range(len(max_indices)):
                if true_labels[i] == max_indices[i]:
                    number_of_correct_labels += 1
                else:
                    number_of_false_labels += 1

            images = torch.cat((images, imgs), dim=0)
            pseudo_labels = torch.cat((pseudo_labels, max_indices))
    
        self.semilogger.info('# No. of correct labels before training with pseudo-labeled data: %d' % (number_of_correct_labels_b))
        self.semilogger.info('# No. of incorrect labels before training with pseudo-labeled data: %d' % (number_of_false_labels_b))
        if number_of_correct_labels_b+number_of_false_labels_b != 0:
                accuracy_b = (number_of_correct_labels_b/(number_of_false_labels_b+number_of_correct_labels_b))
                self.semilogger.info("# Accuracy before training with pseudo-labeled data: %.4f" % accuracy_b)
        else:
            accuracy_b = 0
            self.semilogger.info("# Accuracy before training with pseudo-labeled data: 0")
        self.semilogger.info('# No. of correct labels: %d' % (number_of_correct_labels))
        self.semilogger.info('# No. of incorrect labels: %d' % (number_of_false_labels))
        if number_of_correct_labels+number_of_false_labels != 0:
            accuracy = (number_of_correct_labels/(number_of_false_labels+number_of_correct_labels));
            self.semilogger.info("# Pseudo-labeling accuracy: %.4f" % accuracy)
        else:
            accuracy = 0
            self.semilogger.info("# Pseudo-labeling accuracy: 0")
        self.semilogger.info("# Threshold improvement: %d" % (accuracy_b-accuracy))
        images = images.to("cpu")
        pseudo_labels = pseudo_labels.to("cpu")

        pseudo_labeled_dataset = MXFaceDataset_Split_PseudoLabeled(images, pseudo_labels)
        return pseudo_labeled_dataset

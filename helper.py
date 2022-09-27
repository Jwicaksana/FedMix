import torch
import torchvision
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import os
import copy

from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomCrop
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import adjust_gamma as intensity_shift
import torch.nn as nn

from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg
import numpy as np


from dice_loss import dice_coeff
import random

###############################################
#### CONSTANTS 
###############################################

colors = ['r', 'g', 'b', 'c', 'k', 'y','m', 'c']


#####################################
# Dataloader : ReTouch
# Purpose : for multi classes
#####################################


class retouch(Dataset):
    def __init__(self, data_path, train=False, \
                IMAGE_SIZE=(224,224), CROP_SIZE=(200,200), num_class = 3, \
                noisy=True, augment_intensity=False):
        self.data = data_path
        self.train = train
        self.IMAGE_SIZE = IMAGE_SIZE
        self.CROP_SIZE = CROP_SIZE
        self.NUM_CLASS = num_class
        self.noisy = noisy
        self.augment_intensity=True

    def transform(self, image, mask, train):
        resize_image = Resize(self.IMAGE_SIZE) 
        resize_label = Resize(self.IMAGE_SIZE,  interpolation=TF.InterpolationMode.NEAREST)
        
        image = resize_image(image)
        mask = resize_label(mask)
        
        if train:
            # Random crop
            i, j, h, w = RandomCrop.get_params(
                image, output_size=(self.CROP_SIZE))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
        # Transform to tensor
        image = TF.to_tensor(image)
        
        mask = np.array(mask)
        mask_image_label = np.unique(mask)
        mask = torch.as_tensor(mask, dtype=torch.int64)
        mask = torch.stack([mask==i for i in range(1,self.NUM_CLASS+1)]).type(torch.float)
        return image, mask, mask_image_label
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, label = self.data[idx]
        image = Image.open(data).convert('L')
        mask = Image.open(label)
        x, y, image_label = self.transform(image, mask, self.train)
        # generate bbox mask #####################################
        bbox_mask = torch.zeros(y.shape)
        image_mask = torch.zeros(y.shape)
        
        for item in image_label:
            if item  == 0: #  ignore background
                continue
            tar_interest = copy.deepcopy(y[item-1])
            w, h = torch.where(tar_interest == 1)
            if self.noisy:
                margin = torch.randint(0, 10, (4,))
            else:
                margin = torch.zeros((4,))
            w_min, w_max, h_min, h_max = torch.min(w)-margin[0], \
            torch.max(w)+margin[1], \
            torch.min(h)-margin[2], \
            torch.max(h)+margin[3]
            w_min = torch.clamp(w_min, min=0, max=self.CROP_SIZE[0]).int()
            w_max = torch.clamp(w_max, min=0, max=self.CROP_SIZE[0]).int()
            h_min = torch.clamp(h_min, min=0, max=self.CROP_SIZE[0]).int()
            h_max = torch.clamp(h_max, min=0, max=self.CROP_SIZE[0]).int()
            bbox_mask[item-1, w_min:w_max, h_min:h_max] = 1
            image_mask[item-1] = 1
            
        return x, y, bbox_mask, image_mask
    


#####################################
# Dataloader : Cancer
# Purpose : overall training data single class
#####################################
class Cancer(Dataset):
    def __init__(self, im_path, mask_path, train=False, \
                IMAGE_SIZE=(256,256), CROP_SIZE=(224,224), 
                noisy=True):
        self.data = im_path
        self.label = mask_path
        self.train = train
        self.IMAGE_SIZE = IMAGE_SIZE
        self.CROP_SIZE = CROP_SIZE
        self.noisy = noisy

    def transform(self, image, mask, train):
        resize_image = Resize(self.IMAGE_SIZE)
        resize_label = Resize(self.IMAGE_SIZE, interpolation=TF.InterpolationMode.NEAREST)
        image = resize_image(image)
        mask = resize_label(mask)
        if train:
            # Random crop
            i, j, h, w = RandomCrop.get_params(
                image, output_size=(self.CROP_SIZE))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            # Random vertical flipping
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        return image, mask
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        image = Image.open(self.data[idx]).convert('L')
        mask = Image.open(self.label[idx]).convert('L')
        x, y = self.transform(image, mask, self.train)
        ##########################################################
        # generate bbox mask #####################################
        bbox_mask = torch.zeros(y.shape)
        if torch.sum(y) > 0:
            _, w, h = torch.where(y == 1)
            # should be within the image size,
            if self.noisy:
                margin = torch.randint(1, 10, (4,))
            else:
                margin = torch.zeros((4,))
            w_min, w_max, h_min, h_max = torch.min(w)-margin[0], \
            torch.max(w)+margin[1], \
            torch.min(h)-margin[2], \
            torch.max(h)+margin[3]
            w_min = torch.clamp(w_min, min=0, max=224).int()
            w_max = torch.clamp(w_max, min=0, max=224).int()
            h_min = torch.clamp(h_min, min=0, max=224).int()
            h_max = torch.clamp(h_max, min=0, max=224).int()
            bbox_mask[:, w_min:w_max, h_min:h_max] = 1
        return x, y, bbox_mask


#####################################
# Dataloader : Cancer_v2 or retouch_v2
# Purpose : select pseudo labels 
#####################################
class cancer_v2(Dataset):
    def __init__(self, im_store, pl1_store, pl2_store):
        self.im_store = im_store
        self.pl1_store = pl1_store
        self.pl2_store = pl2_store
        
    def __len__(self):
        return len(self.im_store)
    
    def __getitem__(self, idx):
        x,y1,y2 = self.im_store[idx], self.pl1_store[idx], self.pl2_store[idx]
        return x, y1, y2

class retouch_v2(Dataset):
    def __init__(self, im_store, pl1_store, pl2_store):
        self.im_store = im_store
        self.pl1_store = pl1_store
        self.pl2_store = pl2_store
        
    def __len__(self):
        return len(self.im_store)
    
    def __getitem__(self, idx):
        x,y1,y2 = self.im_store[idx], self.pl1_store[idx], self.pl2_store[idx]
        return x, y1, y2, y2 # second y2 as placeholder


############################################
#### federated aggregation (fedavg) 
#### input: CLIENTS <list of client>
####      : nets <collection of dictionaries>
############################################
def aggr_fed(CLIENTS, WEIGHTS_CL, nets, fed_name='global'):
    for param_tensor in nets[fed_name].state_dict():
        tmp= None
        TOTAL_CLIENTS = len(CLIENTS)
        for client, w in zip(CLIENTS, WEIGHTS_CL):
            if tmp == None:
                tmp = copy.deepcopy(w*nets[client].state_dict()[param_tensor])
            else:
                tmp += w*nets[client].state_dict()[param_tensor]
        nets[fed_name].state_dict()[param_tensor].data.copy_(tmp)
        del tmp


############################################
#### copy federated model to client 
#### input: CLIENTS <list of client>
####      : nets <collection of dictionaries>
############################################
def copy_fed(CLIENTS, nets, fed_name='global'):
    for client in CLIENTS:
        nets[client].load_state_dict(copy.deepcopy(\
            nets[fed_name].state_dict()))    

#############################################
### A helper function to randomly find bbox #
#############################################

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2




# create a specific function to evaluate and select PL #
##########################################
# inputs : networks used for CPS (nets_1, nets_2)
#        : trainloader
#        : array to store the selected images and PL (im_store, pl1, pl2)
#        : TH for selecting the PL and evaluation
#        : bbox and image indicates the type of labels available
# outputs: the dice score of target data and the ratio of data selected [trust]
##########################################
def select_pl_singleclass(nets_1, nets_2, device, trainloader, im_store, \
    pl1_store, pl2_store, \
    TH = 0.9, bbox=False, image=False):
    counter, dice_acc = 0,0 # create variable to store the num data and accuracy
    nets_1.eval()
    nets_2.eval()
    with torch.no_grad():
        # trainloader contains the actual label but is not used # 
        for (imgs, masks, bbox_gt) in trainloader:            
            imgs_cuda1, imgs_cuda2 = imgs.to(device), imgs.to(device)

            y_pred, y2_pred = nets_1(imgs_cuda1), nets_2(imgs_cuda2)
            y_pred, y2_pred = torch.sigmoid(y_pred), torch.sigmoid(y2_pred)
            y_pred, y2_pred = (y_pred > 0.5).float(), (y2_pred > 0.5).float()
            
            if bbox:
                bbox_gt = bbox_gt.to(device)
                y_pred_ = y_pred * bbox_gt
                y2_pred_ = y2_pred * bbox_gt
            
            if bbox:
                dice_net12  = dice_coeff(y2_pred_, y_pred_)
            else:
                dice_net12  = dice_coeff(y2_pred, y_pred)
                
            dice_wrt_gt = dice_coeff(masks.type(torch.float).to(device), y_pred)

            if dice_net12 >= TH:
                dice_acc += dice_wrt_gt
                if bbox:
                    dice_bbox = dice_coeff(y_pred_, bbox_gt)
                    if dice_bbox < 0.5:
                        continue
                im_store.append(imgs[0])
                counter+=1
                if bbox: #refine predictions if bbox
                    y_pred = y_pred_
                    y2_pred = y2_pred_
                pl1_store.append(y_pred[0].detach().cpu())
                pl2_store.append(y2_pred[0].detach().cpu())
        
    # return the counter per total length and dice acc for evaluation
    return counter


def select_pl_multiclasses(nets_1, nets_2, device, trainloader, im_store, \
    pl1_store, pl2_store, \
    TH = 0.9, bbox=False, image=False):
    counter, dice_acc = 0,0 # create variable to store the num data and accuracy
    nets_1.eval()
    nets_2.eval()
    with torch.no_grad():
        # trainloader contains the actual label but is not used # 
        for (imgs, masks, bbox_gt, mask_gt) in trainloader:  
            # bbox_gt, masks, mask_gt dim , (batch, CLASSES [3], W, H)
            imgs_cuda1, imgs_cuda2 = imgs.to(device), imgs.to(device)

            y_pred, y2_pred = nets_1(imgs_cuda1), nets_2(imgs_cuda2)
            y_pred, y2_pred = torch.sigmoid(y_pred), torch.sigmoid(y2_pred)
            y_pred, y2_pred = (y_pred > 0.5).float(), (y2_pred > 0.5).float()
            
            if image:
                mask_gt = mask_gt.to(device)
                y_pred_ = y_pred * mask_gt
                y2_pred_ = y2_pred * mask_gt
            
            if bbox:
                bbox_gt = bbox_gt.to(device)
                y_pred_ = y_pred * bbox_gt
                y2_pred_ = y2_pred * bbox_gt
            
            if bbox or image:
                dice_net12  = dice_coeff(y2_pred_, y_pred_)
            else:
                dice_net12  = dice_coeff(y2_pred, y_pred)
                
            dice_wrt_gt = dice_coeff(masks.type(torch.float).to(device), y_pred)

            if dice_net12 >= TH:
                dice_acc += dice_wrt_gt
                if bbox:
                    dice_bbox = dice_coeff(y_pred_, bbox_gt)
                    if dice_bbox < 0.5:
                        continue

                im_store.append(imgs[0])
                counter+=1
                if bbox or image: #refine predictions
                    y_pred = y_pred_
                    y2_pred = y2_pred_ 

                pl1_store.append(y_pred[0].detach().cpu())
                pl2_store.append(y2_pred[0].detach().cpu())
        
    # return the counter per total length and dice acc for evaluation
    return counter

###########################
## Test the network acc ###
###########################
def test(epoch, testloader, net, device, acc=None, loss=None):
    net.eval()
    t_loss, t_acc = 0,0
    with torch.no_grad():
        for (imgs, masks, _) in testloader:
            masks = masks.type(torch.float32)
            imgs, masks = imgs.to(device), masks.to(device)
            ###########################################
            masks_pred = net(imgs)
            masks_pred = torch.sigmoid(masks_pred)
            l_ = 1 - dice_coeff(masks_pred, masks.type(torch.float))
            t_loss += l_.item()
            #######################################################
            masks_pred = (masks_pred>0.5).float()
            t_acc_network = dice_coeff(masks.type(torch.float), masks_pred).item()
            t_acc += t_acc_network
    
    if acc is not None:
        acc.append(t_acc / len(testloader))
    if loss is not None:
        loss.append(t_loss/ len(testloader))
   
    del t_acc, t_loss

def test_multiclasses(epoch, testloader, net, device, acc=None, loss=None, num_client=3):
    net.eval()
    t_loss, t_acc = 0,0
    with torch.no_grad():
        for (imgs, masks, _,_) in testloader:
            masks = masks.type(torch.float32)
            imgs, masks = imgs.to(device), masks.to(device)
            
            ###########################################
            masks_pred = net(imgs)
            masks_pred = torch.sigmoid(masks_pred)
            l_ = 1 - dice_coeff(masks_pred, masks.type(torch.float))
            t_loss += l_.item()
            #######################################################
            masks_pred = (masks_pred>0.5).float()
            '''
            multiclasses
            '''
            t_acc_network1 = 0.0
            for i in range(num_client):
                t_acc_network1 += dice_coeff(masks[:,i].type(torch.float), \
                                            masks_pred[:,i]).item()
            t_acc_network1 /= 3.0
            t_acc += t_acc_network1
    
    if acc is not None:
        acc.append(t_acc / len(testloader))
    if loss is not None:
        loss.append(t_loss/ len(testloader))
   
    del t_acc, t_loss


'''
Training for every method
if FedST, we augment the image and use crossentropy
'''
# CE_LOSS = nn.BCELoss()

def train_model(trainloader, net_stu, optimizer_stu, \
                     device, acc=None, loss = None, supervision_type='labeled', \
                     warmup=False, CE_LOSS= None, FedMix_network=1):
    net_stu.train()
    t_loss, t_acc = 0,0    
    labeled_len = len(trainloader)
    labeled_iter = iter(trainloader)
    for _ in range(labeled_len):
        imgs, masks, y_pl = next(labeled_iter)
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer_stu.zero_grad()
        ###################################################
        l_ = 0
        ## get the prediction from the model of interest ##
        masks_stu = torch.sigmoid(net_stu(imgs))
        ### if supervision type is labeled, just train as normal with dice ###
        if supervision_type == 'labeled':
            l_stu = (1 - dice_coeff(masks_stu, masks.type(torch.float)))[0]
            l_ = l_stu
        else:
            if warmup: # FedST and FedRGD
                CE_LOSS = CE_LOSS.to(device) 
                # to augment #
                gamma_v = (torch.rand(1)*2)
                gamma_v = gamma_v.to(device)
                imgs_augmented = intensity_shift(imgs, gamma=gamma_v)
                # find confident pixels for training
                mask = (masks_stu>0.9) + (masks_stu<0.1)
                mask = mask.detach()
                masks_stu = (masks_stu.detach() > 0.5).float()

                masks_augmented = torch.sigmoid(net_stu(imgs_augmented))
                l_stu = CE_LOSS(masks_augmented[mask], masks_stu[mask])
                l_ = l_stu

            else: # for non warmup based
                if FedMix_network == 1:
                    masks_teach = y_pl.to(device)
                else:
                    masks_teach = masks.to(device)

                l_stu = (1 - dice_coeff(masks_stu, masks_teach.type(torch.float)))[0]
                l_ = l_stu
        #############################
        l_.backward()
        optimizer_stu.step()

        # for evaluation 
        t_loss += l_.item()
        masks_stu = (masks_stu.detach() > 0.5).float()
        t_acc_network = dice_coeff(masks_stu, masks.type(torch.float)).item()
        t_acc += t_acc_network
                
        
    if acc is not None:
        try:
            acc.append(t_acc/len(trainloader))
        except:
            acc.append(0.0)
    if loss is not None:
        try:
            loss.append(t_loss/len(trainloader))
        except:
            loss.append(0.0)

def train_model_multiclasses(trainloader, net_stu, optimizer_stu, \
                     device, acc=None, loss = None, supervision_type='labeled', \
                     warmup=False, CE_LOSS= None, TH=0.9, FedMix_network=1, num_client=3):
    net_stu.train()
    t_loss, t_acc = 0,0    
    labeled_len = len(trainloader)
    labeled_iter = iter(trainloader)
    for _ in range(labeled_len):
        imgs, masks, y_pl, _ = next(labeled_iter)
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer_stu.zero_grad()
        ###################################################
        l_ = 0
        ## get the prediction from the model of interest ##
        masks_stu = torch.sigmoid(net_stu(imgs))
        ### if supervision type is labeled, just train as normal with dice ###
        if supervision_type == 'labeled':
            dc = dice_coeff(masks_stu, masks.type(torch.float))
            l_stu = (1 - dc)[0]
            l_ = l_stu
        else:
            if warmup:
                CE_LOSS = CE_LOSS.to(device) 
                # to augment #
                gamma_v = (torch.rand(1)*2)
                gamma_v = gamma_v.to(device)
                imgs_augmented = intensity_shift(imgs, gamma=gamma_v)
                # find confident pixels for training
                mask = (masks_stu>TH) + (masks_stu<(1-TH))
                mask = mask.detach()
                masks_stu = (masks_stu.detach() > 0.5).float()

                masks_augmented = torch.sigmoid(net_stu(imgs_augmented))
                masks_augmented = torch.clamp(masks_augmented, min=0, max=1)
                l_stu = CE_LOSS(masks_augmented[mask], masks_stu[mask])
                l_ = l_stu

            else: # for non FedST
                if FedMix_network == 1:
                    masks_teach = y_pl.to(device)
                else:
                    masks_teach = masks.to(device)

                l_stu = (1 - dice_coeff(masks_stu, masks_teach.type(torch.float)))[0]
                l_ = l_stu
        #############################
        l_.backward()
        optimizer_stu.step()

        # for evaluation 
        t_loss += l_.item()
        masks_stu = (masks_stu.detach() > 0.5).float()
        dc = 0.0
        for i in range(num_client):
            dc = dc + dice_coeff(masks_stu[:,i], masks[:,i].type(torch.float))
        dc = dc/3.0
        
        t_acc_network = dc.item()
        t_acc += t_acc_network
                
        
    if acc is not None:
        try:
            acc.append(t_acc/len(trainloader))
        except:
            acc.append(0.0)
    if loss is not None:
        try:
            loss.append(t_loss/len(trainloader))
        except:
            loss.append(0.0)

######################################
####  FUNCTION #######################
######################################






######################################
####  FUNCTION #######################
######################################
######################################
#### plot results 
#### input: num <of plot graph> 
####      : CLIENTS <str list> 
####      : index <x axis > 
####      : y_axis : value 
####      : title : legend 
######################################
def plot_graphs(num, CLIENTS, index, y_axis, title):
    idx_clr = 0
    plt.figure(num)
    for client in CLIENTS:
        plt.plot(index, y_axis[client], colors[idx_clr], label=client+ title)
        idx_clr += 1
    plt.legend()
    plt.show()


########################################
#### save model 
#### input: PTH <saving path>
####      : epoch <identifier>
####      : nets [collection to save]
####      : acc_train : list of clients 
#########################################
def save_model(PTH, epoch, nets, acc_train):
    for client, _ in acc_train.items():
        PATH_MODEL = PTH + client
        os.makedirs(PATH_MODEL, exist_ok=True)
        torch.save(nets[client], PATH_MODEL + '/model_' + str(epoch) +'.pth')

    
def sort_rows(matrix, num_rows):
    matrix_T = torch.transpose(matrix, 0, 1)
    sorted_T = torch.topk(matrix_T, num_rows)[0]
    return torch.transpose(sorted_T, 1, 0)


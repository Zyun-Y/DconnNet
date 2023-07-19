import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch
import numpy as np
from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.io import imread, imsave
import scipy.io as scio


def connectivity_matrix(multimask, class_num):

    ##### converting segmentation masks to connectivity masks ####

    [batch,_,rows, cols] = multimask.shape
    # batch = 1
    conn = torch.zeros([batch,class_num*8,rows, cols]).cuda()
    for i in range(class_num):
        mask = multimask[:,i,:,:]
        # print(mask.shape)
        up = torch.zeros([batch,rows, cols]).cuda()#move the orignal mask to up
        down = torch.zeros([batch,rows, cols]).cuda()
        left = torch.zeros([batch,rows, cols]).cuda()
        right = torch.zeros([batch,rows, cols]).cuda()
        up_left = torch.zeros([batch,rows, cols]).cuda()
        up_right = torch.zeros([batch,rows, cols]).cuda()
        down_left = torch.zeros([batch,rows, cols]).cuda()
        down_right = torch.zeros([batch,rows, cols]).cuda()


        up[:,:rows-1, :] = mask[:,1:rows,:]
        down[:,1:rows,:] = mask[:,0:rows-1,:]
        left[:,:,:cols-1] = mask[:,:,1:cols]
        right[:,:,1:cols] = mask[:,:,:cols-1]
        up_left[:,0:rows-1,0:cols-1] = mask[:,1:rows,1:cols]
        up_right[:,0:rows-1,1:cols] = mask[:,1:rows,0:cols-1]
        down_left[:,1:rows,0:cols-1] = mask[:,0:rows-1,1:cols]
        down_right[:,1:rows,1:cols] = mask[:,0:rows-1,0:cols-1]

        conn[:,(i*8)+0,:,:] = mask*down_right
        conn[:,(i*8)+1,:,:] = mask*down
        conn[:,(i*8)+2,:,:] = mask*down_left
        conn[:,(i*8)+3,:,:] = mask*right
        conn[:,(i*8)+4,:,:] = mask*left
        conn[:,(i*8)+5,:,:] = mask*up_right
        conn[:,(i*8)+6,:,:] = mask*up
        conn[:,(i*8)+7,:,:] = mask*up_left

    conn = conn.float()
    conn = conn.squeeze()
    # print(conn.shape)
    return conn

def Bilateral_voting(c_map,hori_translation,verti_translation):
    
    ####  bilateral voting and convert connectivity-based output into segmentation map. ####

    batch,class_num, channel,row, column = c_map.size()
    vote_out = torch.zeros([batch,class_num,channel, row, column]).cuda()
    # print(vote_out.shape,c_map.shape,hori_translation.shape)
    right = (torch.bmm(c_map[:,:,4].contiguous().view(-1,row,column),hori_translation.view(-1,column,column))).view(batch,class_num,row,column)

    left = (torch.bmm(c_map[:,:,3].contiguous().view(-1,row,column),hori_translation.transpose(3,2).view(-1,column,column))).view(batch,class_num,row,column)

    left_bottom = (torch.bmm(verti_translation.transpose(3,2).view(-1,row,row), c_map[:,:,5].contiguous().view(-1,row,column))).view(batch,class_num,row,column)
    left_bottom = (torch.bmm(left_bottom.view(-1,row,column),hori_translation.transpose(3,2).view(-1,column,column))).view(batch,class_num,row,column)
    right_above = (torch.bmm(verti_translation.view(-1,row,row), c_map[:,:,2].contiguous().view(-1,row,column))).view(batch,class_num,row,column)
    right_above= (torch.bmm(right_above.view(-1,row,column),hori_translation.view(-1,column,column))).view(batch,class_num,row,column)
    left_above = (torch.bmm(verti_translation.view(-1,row,row), c_map[:,:,0].contiguous().view(-1,row,column))).view(batch,class_num,row,column)
    left_above = (torch.bmm(left_above.view(-1,row,column),hori_translation.transpose(3,2).view(-1,column,column))).view(batch,class_num,row,column)
    bottom = (torch.bmm(verti_translation.transpose(3,2).view(-1,row,row), c_map[:,:,6].contiguous().view(-1,row,column))).view(batch,class_num,row,column)
    up = (torch.bmm(verti_translation.view(-1,row,row), c_map[:,:,1].contiguous().view(-1,row,column))).view(batch,class_num,row,column)
    right_bottom = (torch.bmm(verti_translation.transpose(3,2).view(-1,row,row), c_map[:,:,7].contiguous().view(-1,row,column))).view(batch,class_num,row,column)
    right_bottom = (torch.bmm(right_bottom.view(-1,row,column),hori_translation.view(-1,column,column))).view(batch,class_num,row,column)


    vote_out[:,:,0] = (c_map[:,:,0]) * (right_bottom)
    vote_out[:,:,1] = (c_map[:,:,1]) * (bottom )
    vote_out[:,:,2] = (c_map[:,:,2]) * (left_bottom)
    vote_out[:,:,3] = (c_map[:,:,3]) * (right)
    vote_out[:,:,4] = (c_map[:,:,4]) * (left)
    vote_out[:,:,5] = (c_map[:,:,5]) * (right_above)
    vote_out[:,:,6] = (c_map[:,:,6]) * (up)
    vote_out[:,:,7] = (c_map[:,:,7]) * (left_above)

   
    pred_mask,_ = torch.max(vote_out,dim=2)
    ###
    # vote_out = vote_out.view(batch,-1, row, column)
    return pred_mask,vote_out

class dice_loss(nn.Module):
    def __init__(self,bin_wide,density):
        super(dice_loss, self).__init__()
        self.bin_wide = bin_wide
        self.density = density
    def soft_dice_coeff(self, y_pred,y_true,class_i=None):
        smooth = 0.0001  # may change

        i = torch.sum(y_true,dim=(1,2))
        j = torch.sum(y_pred,dim=(1,2))
        intersection = torch.sum(y_true * y_pred,dim=(1,2))

        score = (2. * intersection + smooth) / (i + j + smooth)
        
        if self.bin_wide:
            weight = density_weight(self.bin_wide[class_i], i,self.density[class_i])
            return (1-score)*weight
        else:
            return (1-score)
    def soft_dice_loss(self, y_pred,y_true,class_i=None):
        loss = self.soft_dice_coeff(y_true, y_pred,class_i)
        return loss.mean()

    def __call__(self, y_pred,y_true,class_i=None):

        b = self.soft_dice_loss(y_true, y_pred,class_i)
        return b


def weighted_log_loss(output,weight,target):
    
    #print(weight.min())
    # target = target.type(torch.LongTensor).cuda()
    log_out = F.binary_cross_entropy(output,target,reduction='none')
    # log_out = F.binary_cross_entropy(output,target,reduction='none')
    loss = torch.mean(log_out*weight)
    # print(log_out.shape,weight.shape)

    return loss

def density_weight(bin_wide, gt_cnt,density):

    index = gt_cnt//bin_wide

    selected_density = [density[index[i].long()] for i in range(gt_cnt.shape[0])]
    selected_density = torch.tensor(selected_density).cuda()
    log_inv_density = torch.log(1/(selected_density+0.0001))

    return log_inv_density



class connect_loss(nn.Module):
    def __init__(self,args,hori_translation,verti_translation,density=None,bin_wide=None):
        super(connect_loss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')
        self.BCEloss = nn.BCELoss(reduction='none')
        self.dice_loss = dice_loss(bin_wide=bin_wide,density=density)
        self.bin_wide = bin_wide
        self.density = density
        self.args = args

        self.verti_translation = verti_translation
        self.hori_translation = hori_translation

    def edge_loss(self,vote_out,edge):
        pred_mask_min, _ = torch.min(vote_out.cuda(), dim=2)
        pred_mask_min = pred_mask_min*edge
        minloss = self.BCEloss(pred_mask_min,torch.full_like(pred_mask_min, 0))
        return (minloss.sum()/pred_mask_min.sum())#+maxloss

    def forward(self, c_map, target):
        if self.args.num_class == 1:
            loss = self.single_class_forward(c_map, target)
        else:
            loss = self.multi_class_forward(c_map, target)

        return loss

    def multi_class_forward(self, c_map, target):
        #######
        ### c_map: (B, 8*C, H, W), B: batch, C: class number
        ### target: (B, H, W)
        #######
        target = target.type(torch.LongTensor).cuda()
        batch_num = c_map.shape[0]
        onehotmask = F.one_hot(target.long(),self.args.num_class) # change it to your class number if needed
        onehotmask = onehotmask.permute(0,3,1,2) 
        onehotmask = onehotmask.float()

        ### get your connectivity_mask ###
        con_target = connectivity_matrix(onehotmask,self.args.num_class) #(B, 8*C, H, W)
        
        ### matrix for shifting
        hori_translation = self.hori_translation.repeat(batch_num,1,1,1).cuda()
        verti_translation = self.verti_translation.repeat(batch_num,1,1,1).cuda()
        ## shape for hori_translation: [Batch, NumClass, H, H]
        ## shape for verti_translation: [Batch, NumClass, W, W]
        
        dice_l = 0

        ### get edges gt###
        class_conn = con_target.view([c_map.shape[0],self.args.num_class, 8,c_map.shape[2],c_map.shape[3]])
        sum_conn = torch.sum(class_conn,dim=2)
        ### edge: (B, C, H, W)
        edge = torch.where((sum_conn<8) & (sum_conn>0),torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        
        ### bilateral voting #####
        # final pred: (B, C, H, W), vote_out: (B, C, 8, H, W), bicon_map: (B, 8*C, H, W)
        class_pred = c_map.view([c_map.shape[0],self.args.num_class, 8, c_map.shape[2],c_map.shape[3]])
        final_pred,vote_out = Bilateral_voting(class_pred,hori_translation,verti_translation)
        _, bicon_map = Bilateral_voting(F.sigmoid(class_pred),hori_translation,verti_translation)
        bicon_map = bicon_map.view(con_target.shape)

        edge_l = self.edge_loss(F.sigmoid(vote_out),edge)

        pred = F.softmax(final_pred,dim=1)

        for j in range(1,self.args.num_class):
            dice_l += self.dice_loss(pred[:,j,:,:], onehotmask[:,j],j-1)
        

        ce_loss = F.cross_entropy(final_pred, target)
        conn_l = self.BCEloss(F.sigmoid(c_map),con_target).mean()
        bicon_l = self.BCEloss(bicon_map,con_target).mean()
        loss =  ce_loss + conn_l + edge_l + 0.2* bicon_l+dice_l #+ bce_loss# +loss_out_dice# +sum_l # + edge_l+loss_out_dice

        return loss

    def single_class_forward(self, c_map, target):
        #######
        ### c_map: (B, 8, H, W), B: batch, C: class number
        ### target: (B, 1, H, W)
        #######

        batch_num = c_map.shape[0]
        target = target.float()

        ### get your connectivity_mask ###
        con_target = connectivity_matrix(target,self.args.num_class)#(B, 8, H, W)

        ### matrix for shifting
        hori_translation = self.hori_translation.repeat(batch_num,1,1,1).cuda()
        verti_translation = self.verti_translation.repeat(batch_num,1,1,1).cuda()


        c_map = F.sigmoid(c_map)

        ### get edges gt###
        class_conn = con_target.view([c_map.shape[0],self.args.num_class,8,c_map.shape[2],c_map.shape[3]])
        sum_conn = torch.sum(class_conn,dim=2)
        
        ## edge: (B, 1, H, W)
        edge = torch.where((sum_conn<8) & (sum_conn>0),torch.full_like(sum_conn, 1),torch.full_like(sum_conn, 0))
        
        ### bilateral voting #####
        ## pred: (B, 1, H, W), bicon_map: (B, 1, 8, H, W)
        class_pred = c_map.view([c_map.shape[0],self.args.num_class, 8, c_map.shape[2],c_map.shape[3]])
        pred,bicon_map = Bilateral_voting(class_pred,hori_translation,verti_translation)


        edge_l = self.edge_loss(bicon_map,edge)

        dice_l = self.dice_loss(pred, target)
        

        bce_loss = self.BCEloss(pred, target).mean()
        conn_l = self.BCEloss(c_map,con_target).mean()

        if self.args.dataset == 'chase':
            loss =  bce_loss + conn_l + edge_l +dice_l
        else:
            bicon_l = self.BCEloss(bicon_map.squeeze(1),con_target).mean()
            loss =  bce_loss + conn_l + edge_l + 0.2* bicon_l+dice_l #+ bce_loss# +loss_out_dice# +sum_l # + edge_l+loss_out_dice

        return loss

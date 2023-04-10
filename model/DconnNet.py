# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 09:57:49 2019

@author: Fsl
"""
import math
from model.attention import CAM_Module,PAM_Module
import torch
from torchvision import models
import torch.nn as nn
from model.resnet import resnet34
# from resnet import resnet34
# import resnet
from torch.nn import functional as F
import torchsummary
from torch.nn import init
import model.gap as gap
up_kwargs = {'mode': 'bilinear', 'align_corners': True}


class DconnNet(nn.Module):
    def __init__(self,num_class=1):
        super(DconnNet,self).__init__()
        
        out_planes = num_class*8
        self.backbone =resnet34(pretrained=True)
        self.sde_module = SDE_module(512,512,out_planes)
        self.fb5=FeatureBlock(512,256,relu=False,last=True) #256
        self.fb4=FeatureBlock(256,128,relu=False) #128
        self.fb3=FeatureBlock(128,64,relu=False) #64
        self.fb2=FeatureBlock(64,64) 

        self.gap = gap.GlobalAvgPool2D()

        self.sb1 = SpaceBlock(512,512,512)
        self.sb2 = SpaceBlock(512,256,256)
        self.sb3 = SpaceBlock(256,128,128)
        self.sb4 = SpaceBlock(128,64,64)
        # self.sb5 = SpaceBlock(64,64,32)


        self.relu = nn.ReLU()
        
        self.final_decoder=LWdecoder(in_channels=[64,64,128,256],out_channels=32,in_feat_output_strides=(4, 8, 16, 32),out_feat_output_stride=4,norm_fn=nn.BatchNorm2d,num_groups_gn=None)
        
        self.cls_pred_conv = nn.Conv2d(64, 32, 3,1,1)
        self.cls_pred_conv_2 = nn.Conv2d(32, out_planes, 1)
        self.upsample4x_op = nn.UpsamplingBilinear2d(scale_factor=2)
        self.channel_mapping = nn.Sequential(
                    nn.Conv2d(512, out_planes, 3,1,1),
                    nn.BatchNorm2d(out_planes),
                    nn.ReLU(True)
                )

        self.direc_reencode = nn.Sequential(
                    nn.Conv2d(out_planes, out_planes, 1),
                    # nn.BatchNorm2d(out_planes),
                    # nn.ReLU(True)
                )


    def forward(self, x):
        
        
        
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        c1 = self.backbone.relu(x)#1/2  64
        
        x = self.backbone.maxpool(c1)
        c2 = self.backbone.layer1(x)#1/4   64
        c3 = self.backbone.layer2(c2)#1/8   128
        c4 = self.backbone.layer3(c3)#1/16   256
        c5 = self.backbone.layer4(c4)#1/32   512

        #### directional Prior ####
        directional_c5 = self.channel_mapping(c5)
        mapped_c5=F.interpolate(directional_c5,scale_factor=32,mode='bilinear',align_corners=True)
        mapped_c5 = self.direc_reencode(mapped_c5)
        
        d_prior = self.gap(mapped_c5)

        c5 = self.sde_module(c5,d_prior)


        c6 = self.gap(c5)
        

        r5 = self.sb1(c6,c5)

        d4=self.relu(self.fb5(r5)+c4)  #256
        r4 = self.sb2(self.gap(r5),d4) 

        d3=self.relu(self.fb4(r4)+c3)  #128
        r3 = self.sb3(self.gap(r4),d3) 

        d2=self.relu(self.fb3(r3)+c2) #64
        r2 = self.sb4(self.gap(r3),d2) 

        d1=self.fb2(r2)+c1 #32
        # d1 = self.sr5(c6,d1) 

        feat_list = [d1,d2,d3,d4,c5]


        final_feat = self.final_decoder(feat_list)

        cls_pred = self.cls_pred_conv_2(final_feat)
        cls_pred = self.upsample4x_op(cls_pred)

        return cls_pred,mapped_c5
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
                # m.weight.data.fill_(1)
                # m.bias.data.zero_()
#        return F.logsigmoid(main_out,dim=1)


class SDE_module(nn.Module):
    def __init__(self, in_channels, out_channels, num_class):
        super(SDE_module, self).__init__()
        self.inter_channels = in_channels // 8

        self.att1 = DANetHead(self.inter_channels,self.inter_channels)
        self.att2 = DANetHead(self.inter_channels,self.inter_channels)
        self.att3 = DANetHead(self.inter_channels,self.inter_channels)
        self.att4 = DANetHead(self.inter_channels,self.inter_channels)
        self.att5 = DANetHead(self.inter_channels,self.inter_channels)
        self.att6 = DANetHead(self.inter_channels,self.inter_channels)
        self.att7 = DANetHead(self.inter_channels,self.inter_channels)
        self.att8 = DANetHead(self.inter_channels,self.inter_channels)


        self.final_conv = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, out_channels, 1))
        #self.encoder_block = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(in_channels, 32, 1))
        
        if num_class<32:
            self.reencoder = nn.Sequential(
                        nn.Conv2d(num_class, num_class*8, 1),
                        nn.ReLU(True),
                        nn.Conv2d(num_class*8, in_channels, 1))
        else:
            self.reencoder = nn.Sequential(
                        nn.Conv2d(num_class, in_channels, 1),
                        nn.ReLU(True),
                        nn.Conv2d(in_channels, in_channels, 1))

    def forward(self, x, d_prior):

        ### re-order encoded_c5 ###
        # new_order = [0,8,16,24,1,9,17,25,2,10,18,26,3,11,19,27,4,12,20,28,5,13,21,29,6,14,22,30,7,15,23,31]
        # # print(encoded_c5.shape)
        # re_order_d_prior = d_prior[:,new_order,:,:]
        # print(d_prior)
        enc_feat = self.reencoder(d_prior)

        feat1 = self.att1(x[:,:self.inter_channels], enc_feat[:,0:self.inter_channels])
        feat2 = self.att2(x[:,self.inter_channels:2*self.inter_channels],enc_feat[:,self.inter_channels:2*self.inter_channels])
        feat3 = self.att3(x[:,2*self.inter_channels:3*self.inter_channels],enc_feat[:,2*self.inter_channels:3*self.inter_channels])
        feat4 = self.att4(x[:,3*self.inter_channels:4*self.inter_channels],enc_feat[:,3*self.inter_channels:4*self.inter_channels])
        feat5 = self.att5(x[:,4*self.inter_channels:5*self.inter_channels],enc_feat[:,4*self.inter_channels:5*self.inter_channels])
        feat6 = self.att6(x[:,5*self.inter_channels:6*self.inter_channels],enc_feat[:,5*self.inter_channels:6*self.inter_channels])
        feat7 = self.att7(x[:,6*self.inter_channels:7*self.inter_channels],enc_feat[:,6*self.inter_channels:7*self.inter_channels])
        feat8 = self.att8(x[:,7*self.inter_channels:8*self.inter_channels],enc_feat[:,7*self.inter_channels:8*self.inter_channels])
        
        feat = torch.cat([feat1,feat2,feat3,feat4,feat5,feat6,feat7,feat8],dim=1)

        sasc_output = self.final_conv(feat)
        sasc_output = sasc_output+x

        return sasc_output


class DANetHead(nn.Module):
    def __init__(self, in_channels, inter_channels, norm_layer=nn.BatchNorm2d):
        super(DANetHead, self).__init__()
        # inter_channels = in_channels // 8
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        
        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)
        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        # self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        # self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, inter_channels, 1))

    def forward(self, x, enc_feat):
        feat1 = self.conv5a(x)
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        # sa_output = self.conv6(sa_conv)
        
        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        # sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv
        
        feat_sum = feat_sum*F.sigmoid(enc_feat)

        sasc_output = self.conv8(feat_sum)


        return sasc_output




class SpaceBlock(nn.Module):
    def __init__(self,
                 in_channels,
                 channel_in,
                 out_channels,
                 scale_aware_proj=False):
        super(SpaceBlock, self).__init__()
        self.scale_aware_proj = scale_aware_proj


        self.scene_encoder = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 1),
        )

        self.content_encoders=nn.Sequential(
                nn.Conv2d(channel_in, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        
        self.feature_reencoders=nn.Sequential(
                nn.Conv2d(channel_in, out_channels, 1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(True)
            )
        

        self.normalizer = nn.Sigmoid()

    def forward(self, scene_feature, features):
        content_feats = self.content_encoders(features)

        scene_feat = self.scene_encoder(scene_feature)
        relations = self.normalizer((scene_feat * content_feats).sum(dim=1, keepdim=True))

        p_feats = self.feature_reencoders(features) 

        refined_feats = relations * p_feats 

        return refined_feats






class LWdecoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_feat_output_strides=(4, 8, 16, 32),
                 out_feat_output_stride=4,
                 norm_fn=nn.BatchNorm2d,
                 num_groups_gn=None):
        super(LWdecoder, self).__init__()
        if norm_fn == nn.BatchNorm2d:
            norm_fn_args = dict(num_features=out_channels)
        elif norm_fn == nn.GroupNorm:
            if num_groups_gn is None:
                raise ValueError('When norm_fn is nn.GroupNorm, num_groups_gn is needed.')
            norm_fn_args = dict(num_groups=num_groups_gn, num_channels=out_channels)
        else:
            raise ValueError('Type of {} is not support.'.format(type(norm_fn)))
        self.blocks = nn.ModuleList()
        dec_level = 0
        for in_feat_os in in_feat_output_strides:
            num_upsample = int(math.log2(int(in_feat_os))) - int(math.log2(int(out_feat_output_stride)))

            num_layers = num_upsample if num_upsample != 0 else 1

            self.blocks.append(nn.Sequential(*[
                nn.Sequential(
                    nn.Conv2d(in_channels[dec_level] if idx ==0 else out_channels, out_channels, 3, 1, 1, bias=False),
                    norm_fn(**norm_fn_args) if norm_fn is not None else nn.Identity(),
                    nn.ReLU(inplace=True),
                    nn.UpsamplingBilinear2d(scale_factor=2) if num_upsample != 0 else nn.Identity(),
                )
                for idx in range(num_layers)]))
            dec_level+=1

    def forward(self, feat_list: list):
        inner_feat_list = []
        for idx, block in enumerate(self.blocks):
            decoder_feat = block(feat_list[idx])
            inner_feat_list.append(decoder_feat)

        out_feat = sum(inner_feat_list) / 4.
        return out_feat

class FeatureBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 norm_layer=nn.BatchNorm2d,scale=2,relu=True,last=False):
        super(FeatureBlock, self).__init__()
       

        self.conv_3x3 = ConvBnRelu(in_planes, in_planes, 3, 1, 1,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
        self.conv_1x1 = ConvBnRelu(in_planes, out_planes, 1, 1, 0,
                                   has_bn=True, norm_layer=norm_layer,
                                   has_relu=True, has_bias=False)
       
        self.scale=scale
        self.last=last

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):

        if self.last==False:
            x = self.conv_3x3(x)
        if self.scale>1:
            x=F.interpolate(x,scale_factor=self.scale,mode='bilinear',align_corners=True)
        x=self.conv_1x1(x)
        return x


    
class ConvBnRelu(nn.Module):
    def __init__(self, in_planes, out_planes, ksize, stride, pad, dilation=1,
                 groups=1, has_bn=True, norm_layer=nn.BatchNorm2d,
                 has_relu=True, inplace=True, has_bias=False):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=ksize,
                              stride=stride, padding=pad,
                              dilation=dilation, groups=groups, bias=has_bias)
        self.has_bn = has_bn
        if self.has_bn:
            self.bn = nn.BatchNorm2d(out_planes)
        self.has_relu = has_relu
        if self.has_relu:
            self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        if self.has_bn:
            x = self.bn(x)
        if self.has_relu:
            x = self.relu(x)

        return x
    


    

# if __name__ == '__main__':
   

#    model = DconnNet()
#    torchsummary.summary(model, (3, 512, 512))

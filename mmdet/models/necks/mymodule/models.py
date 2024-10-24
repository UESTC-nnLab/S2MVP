#!/usr/bin/env python
# coding: utf-8
#
# Author:   Pengxiang Yan
# Email:    yanpx (at) mail2.sysu.edu.cn

from .rcrnet import RCRNet
from .convgru import ConvGRUCell
from .non_local_dot_product import NONLocalBlock3D

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageModel(nn.Module):
    '''
        RCRNet
    '''
    def __init__(self, pretrained=False):
        super(ImageModel, self).__init__()
        self.backbone = RCRNet(
            n_classes=1,
            output_stride=16,
            pretrained=pretrained
        )
    def forward(self, frame):
        seg = self.backbone(frame)
        return seg

class VideoModel(nn.Module):
    '''
        RCRNet+NER
    '''
    def __init__(self, output_stride=16):
        super(VideoModel, self).__init__()
        # video mode + video dataset
        self.backbone = RCRNet(
            n_classes=1,
            output_stride=output_stride,
            pretrained=False,
            input_channels=3
        )
        self.convgru_forward = ConvGRUCell(256, 256, 3)
        self.convgru_backward = ConvGRUCell(256, 256, 3)
        self.bidirection_conv = nn.Conv2d(512, 256, 3, 1, 1)

        self.non_local_block = NONLocalBlock3D(256, sub_sample=False, bn_layer=False)
        self.non_local_block2 = NONLocalBlock3D(256, sub_sample=False, bn_layer=False)

        self.freeze_bn()

    def freeze_bn(self):
        for m in self.backbone.named_modules():
            if isinstance(m[1], nn.BatchNorm2d):
                m[1].eval()

    def forward(self, clip):
        clip_feats = [self.backbone.feat_conv(frame) for frame in clip] # {[1,256,128,128] [1,512,64,64] [1,1024,32,32],[1,256,32,32]}*5
        feats_time = [feats[-1] for feats in clip_feats] #把每个clip的最后一层特征拿出来
        feats_time = torch.stack(feats_time, dim=2) # [1,256,5,32,32]
        feats_time = self.non_local_block(feats_time) # [1,256,5,32,32]

        # Deep Bidirectional ConvGRU
        frame = clip[0] # [1,3,512,512] 第一帧图像特征
        feat = feats_time[:,:,0,:,:] # [1,256,32,32] 第一帧特征
        feats_forward = []
        # forward
        for i in range(len(clip)):
            feat = self.convgru_forward(feats_time[:,:,i,:,:], feat) # [1,256,32,32]
            feats_forward.append(feat)
        # backward
        feat = feats_forward[-1] #最后一帧特征
        feats_backward = []
        for i in range(len(clip)):
            feat = self.convgru_backward(feats_forward[len(clip)-1-i], feat)
            feats_backward.append(feat)

        feats_backward = feats_backward[::-1] #反转，让最后一帧还在最后
        feats = []
        for i in range(len(clip)):
            feat = torch.tanh(self.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
            feats.append(feat)
        feats = torch.stack(feats, dim=2) # [1,256,5,32,32]

        feats = self.non_local_block2(feats)

        
        preds = []
        for i, frame in enumerate(clip):
            seg = self.backbone.seg_conv(clip_feats[i][0], clip_feats[i][1], clip_feats[i][2], feats[:,:,i,:,:], frame.shape[2:])
            preds.append(seg)
        return preds

#测试video model
if __name__ == '__main__':
    model = VideoModel()
    img = torch.randn(1, 3, 512, 512)
    clip = [img, img, img,img,img]
    out = model(clip)
    print(out)
    print(out[0].shape)
# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import torch.nn as nn
from mmcv.cnn import ConvModule
from mmengine.model import BaseModule
from torch import Tensor
from mmdet.registry import MODELS
from mmdet.utils import OptConfigType, OptMultiConfig
import pdb
import torch
import torch.nn.functional as F
import torch.nn as nn
@MODELS.register_module()
class ChannelMapper(BaseModule):
    """Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            convolution layer. Default: None.
        norm_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            normalization layer. Default: None.
        act_cfg (:obj:`ConfigDict` or dict, optional): Config dict for
            activation layer in ConvModule. Default: dict(type='ReLU').
        bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        num_outs (int, optional): Number of output feature maps. There would
            be extra_convs when num_outs larger than the length of in_channels.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or dict],
            optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int,
        kernel_size: int = 3,
        conv_cfg: OptConfigType = None,
        norm_cfg: OptConfigType = None,
        act_cfg: OptConfigType = dict(type='ReLU'),
        bias: Union[bool, str] = 'auto',
        num_outs: int = None,
        init_cfg: OptMultiConfig = dict(
            type='Xavier', layer='Conv2d', distribution='uniform')
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        if num_outs is None:
            num_outs = len(in_channels)
        self.convs = nn.ModuleList()
        for in_channel in in_channels:
            self.convs.append(
                ConvModule(
                    in_channel,
                    out_channels,
                    kernel_size,
                    padding=(kernel_size - 1) // 2,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=bias))
        if num_outs > len(in_channels):
            self.extra_convs = nn.ModuleList()
            for i in range(len(in_channels), num_outs):
                if i == len(in_channels):
                    in_channel = in_channels[-1]
                else:
                    in_channel = out_channels
                self.extra_convs.append(
                    ConvModule(
                        in_channel,
                        out_channels,
                        3,
                        stride=2,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=act_cfg,
                        bias=bias))

    def forward(self, inputs: Tuple[Tensor]) -> Tuple[Tensor]:
        """Forward function."""
        assert len(inputs) == len(self.convs)
        outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        if self.extra_convs:
            for i in range(len(self.extra_convs)):
                if i == 0:
                    outs.append(self.extra_convs[0](inputs[-1]))
                else:
                    outs.append(self.extra_convs[i](outs[-1]))
        return tuple(outs)




from .mymodule.darknet import BaseConv
from .mymodule.rcrnet import _ASPPModule
from .mymodule.non_local_dot_product import NONLocalBlock3D
from .mymodule.convgru import ConvGRUCell

@MODELS.register_module()
class SeqNeck(nn.Module):
    def __init__(self,stride=16,output_channel=256,num_frame=5):
        super(SeqNeck, self).__init__()
        self.num_frame = num_frame
        self.aspp = _ASPPModule(2048, output_channel, stride)
        self.convgru_forward = ConvGRUCell(output_channel, output_channel, 3)
        self.convgru_backward = ConvGRUCell(output_channel, output_channel, 3)
        self.bidirection_conv = nn.Conv2d(output_channel*2, output_channel, 3, 1, 1)
        # self.bidirection_conv2 = nn.Sequential(
        #     BaseConv(512, 256,3,1),
        #     BaseConv(256,256,3,1)
        # )

        self.non_local_block = NONLocalBlock3D(output_channel, sub_sample=False, bn_layer=False)
        self.non_local_block2 = NONLocalBlock3D(output_channel, sub_sample=False, bn_layer=False)

        
        self.conv_ref = nn.Sequential(
            BaseConv(output_channel*(self.num_frame-1), output_channel*2,3,1),
            BaseConv(output_channel*2,output_channel,3,1, act='sigmoid')
        )
        self.conv_cur = BaseConv(output_channel, output_channel,3,1)
        self.conv_cr_mix = nn.Sequential(
            BaseConv(output_channel*2, output_channel*2,3,1),
            BaseConv(output_channel*2,output_channel,3,1)
        )

    def forward(self, inputs): #inputs: [[(1,512,64,64),(1,1024,32,32),(1,2048,16,16)],[],[],[],[]]
        # pdb.set_trace()
        num_frame = len(inputs)
        ####监督的batchsize=1不能通过aspp    clip_feats = [self.aspp(frame[-1]) for frame in inputs]
        temp = []
        for frame in inputs:
            temp.append(frame[-1])
        if inputs[0][0].shape[0] == 1:
            temp = torch.cat(temp, dim=0) #torch.Size([5, 2048, 32, 32])
            temps = self.aspp(temp)
            clip_feats = torch.chunk(temps, num_frame, dim=0)
        else:
            clip_feats = [self.aspp(frame[-1]) for frame in inputs]
        # for i in range(len(inputs)):
        #     clip_feats.append(temps[i,:,:,:].unsqueeze(0))
        

        feats_time = torch.stack(clip_feats, dim=2) # [1,256,5,32,32]
        # feats_time = self.non_local_block(feats_time) # [1,256,5,32,32]
        
        # Deep Bidirectional ConvGRU
        feat = feats_time[:,:,0,:,:] # [1,256,32,32] 第一帧特征
        feats_forward = []
         # forward
        for i in range(len(inputs)):
            feat = self.convgru_forward(feats_time[:,:,i,:,:], feat) # [1,256,32,32]
            feats_forward.append(feat)
        
        # backward
        feat = feats_forward[-1] #最后一帧特征
        feats_backward = []
        for i in range(len(inputs)):
            feat = self.convgru_backward(feats_forward[len(inputs)-1-i], feat)
            feats_backward.append(feat)
        feats_backward = feats_backward[::-1] #反转，让最后一帧还在最后
        
        feats = []
        for i in range(len(inputs)):
            feat = torch.tanh(self.bidirection_conv(torch.cat((feats_forward[i], feats_backward[i]), dim=1)))
            feats.append(feat)
        feats = torch.stack(feats, dim=2) # [1,256,5,32,32]
        
        # feats = self.non_local_block2(feats) # [1,256,5,32,32]

        res = []
        for i in range(len(inputs)):
            res.append(feats[:,:,i,:,:])
        
        return res
        
        # rc_feat = torch.cat([res[j] for j in range(len(inputs)-1)],dim=1)  # 参考帧在通道维度融合
        # r_feat = self.conv_ref(rc_feat)  #通过sigmoid计算权重
        # c_feat = self.conv_cur(r_feat*res[-1]) #和关键帧相乘
        # c_feat = self.conv_cr_mix(torch.cat([c_feat, res[-1]], dim=1)) #4,256，32,32
        # results = []
        # for i in range(5):
        #     results.append(c_feat) #[1,256,32,32]
        # return results


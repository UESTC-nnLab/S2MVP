# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector

import warnings
import torch
import torch.nn as nn
from torch.nn import functional as F

class SRNet(nn.Module):
    """
    Implementation based on methods from the AIM 2022 Challenge on
    Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs
    https://arxiv.org/pdf/2211.05910.pdf
    """

    def __init__(self,
                 num_channels,
                 num_feats,
                 num_blocks,
                 upscale) -> None:
        super(SRNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(num_channels, num_feats, 3, padding=1)
        )

        body = []
        for i in range(num_blocks):
            body.append(nn.Conv2d(num_feats, num_feats, 3, padding=1))
            body.append(nn.ReLU(True))

        self.body = nn.Sequential(*body)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, 3 * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        res = self.head(x)
        out = self.body(res)
        out = self.upsample(res + out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm + relu"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)



class BasicIRNet(nn.Module):
    """
    Implementation based on methods from the AIM 2022 Challenge on
    Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs
    https://arxiv.org/pdf/2211.05910.pdf
    """

    def __init__(self,
                 in_plane,
                 upscale) -> None:
        super(BasicIRNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_plane, in_plane, 3, padding=1)
        )

        self.body = nn.ModuleList()
        self.num_upsample = 2 if upscale is 4 else 3
        for i in range(self.num_upsample):
            self.body.append(conv3x3(int(in_plane/2**i), int(in_plane / 2**(i+1))))

        self.end = nn.Conv2d(int(in_plane / 2**(self.num_upsample)), 2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

    def forward(self, x):

        x = self.head(x)
        for i in range(self.num_upsample):
            x = resize(self.body[i](x), scale_factor=(2, 2), mode='bilinear')
        out = self.end(x)
        return out


class HRFE(nn.Module):
    """
    A high resolution feature enhancement module for tiny object detection
    """

    def __init__(self,
                 in_channels,
                 num_blocks) -> None:
        """
        Args:
            in_channels: the channel of input feature map
            num_blocks: the nums of hrfe module
        """
        super(HRFE, self).__init__()

        body = []
        for i in range(num_blocks):
            body.append(nn.Conv2d(in_channels, in_channels, 3, padding=1))
            body.append(nn.BatchNorm2d(in_channels))
            body.append(nn.ReLU(True))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        out = self.body(x)
        return out

class MRFAFE(nn.Module):
    def __init__(self, in_channels, group, kernel_sizes=(3, 7, 21)):
        super(MRFAFE, self).__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[0],
                                               padding=kernel_sizes[0]//2),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())
        self.branch2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[1],
                                               padding=kernel_sizes[1]//2, groups=in_channels),
                                     nn.BatchNorm2d(in_channels),
                                     nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[1],
                                               padding=kernel_sizes[1]//2, groups=in_channels),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())
        self.branch3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[2],
                                               padding=kernel_sizes[2] // 2, groups=in_channels),
                                     nn.BatchNorm2d(in_channels),
                                     nn.Conv2d(in_channels, in_channels, kernel_size=kernel_sizes[2],
                                               padding=kernel_sizes[2] // 2, groups=in_channels),
                                     nn.BatchNorm2d(in_channels),
                                     nn.ReLU())

        self.conv1 = nn.Conv2d(in_channels, 1, kernel_size=1)

        self.weight_module = Weight(3 * in_channels, group) 
        self.in_channels = in_channels

    def forward(self, x):
        res = x
        x = torch.cat((self.branch1(x), self.branch2(x), self.branch3(x)), dim=1)
        weight = self.weight_module(x)
        x = x * weight
        x = x[:, 0:self.in_channels] + x[:, self.in_channels:2*self.in_channels] + \
            x[:, 2*self.in_channels:3*self.in_channels]
        x = self.conv1(x)
        return x + res

class NonLocalBlock(nn.Module):
    def __init__(self, planes, reduce_ratio=8):
        super(NonLocalBlock, self).__init__()

        inter_planes = planes // reduce_ratio
        self.query_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.key_conv = nn.Conv2d(planes, inter_planes, kernel_size=1)
        self.value_conv = nn.Conv2d(planes, planes, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()

        proj_query = self.query_conv(x)
        proj_key = self.key_conv(x)
        proj_value = self.value_conv(x)

        proj_query = proj_query.contiguous().view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = proj_key.contiguous().view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = proj_value.contiguous().view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, -1, height, width)

        out = self.gamma * out + x
        return out

class Weight(nn.Module):
    def __init__(self, in_channels, group):
        super(Weight, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=group, kernel_size=(3, 3),
                                            padding=1, groups=group))
        self.softmax = nn.Softmax(dim=1)
        self.group = group
        self.repeat = int(in_channels/group)

    def forward(self, x):
        x = self.conv(x)
        x = torch.sum(x, (2, 3), keepdim=True)
        weight = self.softmax(x)
        weight = weight.repeat(1, self.repeat, 1, 1)
        weight = torch.cat(tuple((weight[:, i::self.group, :, :] for i in range(self.group))), dim=1)
        return weight



import pdb
@MODELS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 bbox_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 add_seg = False,
                 add_enhance = False,
                 weight_or=10,
                 ) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        
        # pdb.set_trace() 
        if add_seg:
            self.add_seg = True
            # self.branch_ir = self.build_ir(num_channels=256, num_feats=48, upscale=bbox_head['strides'][0], num_blocks=1)

            if bbox_head['type'] == 'RepPointsHead':
                self.branch_ir = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['point_strides'][0])
            elif bbox_head['type'] == 'FCOSHead':
                self.branch_ir = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['strides'][0])
            elif bbox_head['type'] == 'ATSSHead':
                self.branch_ir = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['anchor_generator']['strides'][0])
            else:
                self.branch_ir = BasicIRNet(in_plane=neck['out_channels'], upscale=8)

            self.loss_or = nn.CrossEntropyLoss()
            self.weight_or = weight_or
            self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.up4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
            self.up8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
            self.up16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)

        if add_enhance:
            self.add_enhance = True
            # self.branch_hrfe = MRFAFE(256, 3)
            self.enhance = NonLocalBlock(256)



    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading two-stage
        weights into single-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
            for rpn_head_key in rpn_head_keys:
                bbox_head_key = bbox_head_prefix + \
                                rpn_head_key[len(rpn_head_prefix):]
                state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)  
        losses = self.bbox_head.loss(x, batch_data_samples)
        #####
        # pdb.set_trace()
        if hasattr(self, 'add_seg'):
            loss = dict()
            object_maps = self.build_target_obj(batch_data_samples, batch_data_samples)
            loss_or0 = self.loss_reconstruction(x[0], object_maps)
            loss_or1 = self.loss_reconstruction(self.up2(x[1]), object_maps)
            loss_or2 = self.loss_reconstruction(self.up4(x[2]), object_maps)
            loss_or3 = self.loss_reconstruction(self.up8(x[3]), object_maps)
            # loss_or4 = self.loss_reconstruction(self.up16(x[4]), object_maps)
            loss_or = loss_or0 + loss_or1 + loss_or2 + loss_or3 #+ loss_or4
            loss['loss_or'] = loss_or
            losses.update(loss)

        return losses,x[0]

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances'. And the ``pred_instances`` usually
            contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
        """
        x = self.extract_feat(batch_inputs)
        results_list = self.bbox_head.predict(
            x, batch_data_samples, rescale=rescale)
        batch_data_samples = self.add_pred_to_datasample(
            batch_data_samples, results_list)
        return batch_data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x = self.extract_feat(batch_inputs)
        results = self.bbox_head.forward(x)
        return results

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)

        #####
        # pdb.set_trace()
        if hasattr(self, 'add_enhance'):
            x0 = x[0]
            x_ = x[1:]
            # x0 = self.branch_hrfe(x0)
            x0 = self.enhance(x0)
            x = [x0,] + x_
        return x

    # 构建目标mask
    def build_target_obj(self, samples, img_metas):
        # build object map
        list_object_maps = []
        for i, sample in enumerate(samples):
            object_map = torch.zeros(img_metas[0].batch_input_shape).to(sample.gt_instances['bboxes'].device)
            gt_bbox = sample.gt_instances['bboxes']
            for index in range(gt_bbox.shape[0]):
                gt = gt_bbox[index]
                # 宽和高都小于64为条件
                if (int(gt[2])-int(gt[0])) <= 64 and (int(gt[3]) - int(gt[1])) <= 64:
                    object_map[int(gt[1]):(int(gt[3])+1), int(gt[0]):(int(gt[2])+1)] = 1

            list_object_maps.append(object_map[None])

        object_maps = torch.cat(list_object_maps, dim=0)
        return object_maps.long()
  
    # def build_target_obj(self, gt_bboxes, img_metas):
    #     # build object map
    #     list_object_maps = []
    #     for i, gt_bbox in enumerate(gt_bboxes):
    #         object_map = torch.zeros(img_metas[0]["batch_input_shape"], device=gt_bboxes[0].device)
    #         for index in range(gt_bbox.shape[0]):
    #             gt = gt_bbox[index]
    #             # 宽和高都小于64为条件
    #             if (int(gt[2])-int(gt[0])) <= 64 and (int(gt[3]) - int(gt[1])) <= 64:
    #                 object_map[int(gt[1]):(int(gt[3])+1), int(gt[0]):(int(gt[2])+1)] = 1

    #         list_object_maps.append(object_map[None])

    #     object_maps = torch.cat(list_object_maps, dim=0)
    #     return object_maps.long()


    # TODO：define a construction image function by ldy
    def loss_reconstruction(self, x, img):
        """
        Args:
            x (Tensor): the frature map used for reconstruction img
            img (Tensor): Input images of shape (N, C, H, W).
        Returns:
            dict[str, Tensor]: A dictionary of reconstruction loss.
        """
        # pdb.set_trace()
        loss = dict()
        x = self.branch_ir(x)
        loss_rec = self.weight_or * self.loss_or(x, img)
        # loss['loss_or'] = loss_rec
        return loss_rec



####original
# @MODELS.register_module()
# class SingleStageDetector(BaseDetector):
#     """Base class for single-stage detectors.

#     Single-stage detectors directly and densely predict bounding boxes on the
#     output features of the backbone+neck.
#     """

#     def __init__(self,
#                  backbone: ConfigType,
#                  neck: OptConfigType = None,
#                  bbox_head: OptConfigType = None,
#                  train_cfg: OptConfigType = None,
#                  test_cfg: OptConfigType = None,
#                  data_preprocessor: OptConfigType = None,
#                  init_cfg: OptMultiConfig = None) -> None:
#         super().__init__(
#             data_preprocessor=data_preprocessor, init_cfg=init_cfg)
#         self.backbone = MODELS.build(backbone)
#         if neck is not None:
#             self.neck = MODELS.build(neck)
#         bbox_head.update(train_cfg=train_cfg)
#         bbox_head.update(test_cfg=test_cfg)
#         self.bbox_head = MODELS.build(bbox_head)
#         self.train_cfg = train_cfg
#         self.test_cfg = test_cfg

#     def _load_from_state_dict(self, state_dict: dict, prefix: str,
#                               local_metadata: dict, strict: bool,
#                               missing_keys: Union[List[str], str],
#                               unexpected_keys: Union[List[str], str],
#                               error_msgs: Union[List[str], str]) -> None:
#         """Exchange bbox_head key to rpn_head key when loading two-stage
#         weights into single-stage model."""
#         bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
#         bbox_head_keys = [
#             k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
#         ]
#         rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
#         rpn_head_keys = [
#             k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
#         ]
#         if len(bbox_head_keys) == 0 and len(rpn_head_keys) != 0:
#             for rpn_head_key in rpn_head_keys:
#                 bbox_head_key = bbox_head_prefix + \
#                                 rpn_head_key[len(rpn_head_prefix):]
#                 state_dict[bbox_head_key] = state_dict.pop(rpn_head_key)
#         super()._load_from_state_dict(state_dict, prefix, local_metadata,
#                                       strict, missing_keys, unexpected_keys,
#                                       error_msgs)

#     def loss(self, batch_inputs: Tensor,
#              batch_data_samples: SampleList) -> Union[dict, list]:
#         """Calculate losses from a batch of inputs and data samples.

#         Args:
#             batch_inputs (Tensor): Input images of shape (N, C, H, W).
#                 These should usually be mean centered and std scaled.
#             batch_data_samples (list[:obj:`DetDataSample`]): The batch
#                 data samples. It usually includes information such
#                 as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

#         Returns:
#             dict: A dictionary of loss components.
#         """
#         x = self.extract_feat(batch_inputs)
#         losses = self.bbox_head.loss(x, batch_data_samples)
#         return losses

#     def predict(self,
#                 batch_inputs: Tensor,
#                 batch_data_samples: SampleList,
#                 rescale: bool = True) -> SampleList:
#         """Predict results from a batch of inputs and data samples with post-
#         processing.

#         Args:
#             batch_inputs (Tensor): Inputs with shape (N, C, H, W).
#             batch_data_samples (List[:obj:`DetDataSample`]): The Data
#                 Samples. It usually includes information such as
#                 `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
#             rescale (bool): Whether to rescale the results.
#                 Defaults to True.

#         Returns:
#             list[:obj:`DetDataSample`]: Detection results of the
#             input images. Each DetDataSample usually contain
#             'pred_instances'. And the ``pred_instances`` usually
#             contains following keys.

#                 - scores (Tensor): Classification scores, has a shape
#                     (num_instance, )
#                 - labels (Tensor): Labels of bboxes, has a shape
#                     (num_instances, ).
#                 - bboxes (Tensor): Has a shape (num_instances, 4),
#                     the last dimension 4 arrange as (x1, y1, x2, y2).
#         """
#         x = self.extract_feat(batch_inputs)
#         results_list = self.bbox_head.predict(
#             x, batch_data_samples, rescale=rescale)
#         batch_data_samples = self.add_pred_to_datasample(
#             batch_data_samples, results_list)
#         return batch_data_samples

#     def _forward(
#             self,
#             batch_inputs: Tensor,
#             batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
#         """Network forward process. Usually includes backbone, neck and head
#         forward without any post-processing.

#          Args:
#             batch_inputs (Tensor): Inputs with shape (N, C, H, W).
#             batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
#                 the meta information of each image and corresponding
#                 annotations.

#         Returns:
#             tuple[list]: A tuple of features from ``bbox_head`` forward.
#         """
#         x = self.extract_feat(batch_inputs)
#         results = self.bbox_head.forward(x)
#         return results

#     def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
#         """Extract features.

#         Args:
#             batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

#         Returns:
#             tuple[Tensor]: Multi-level features that may have
#             different resolutions.
#         """
#         x = self.backbone(batch_inputs)
#         if self.with_neck:
#             x = self.neck(x)
#         return x

# Copyright (c) OpenMMLab. All rights reserved.
import copy
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor 
import torch.nn.functional as F

from mmdet.models.utils import (filter_gt_instances, rename_loss_dict,
                                reweight_loss_dict)
from mmdet.registry import MODELS
from mmdet.visualization import DetLocalVisualizer
from mmdet.structures import DetDataSample, SampleList
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig
from mmdet.models.detectors import SemiBaseDetector
from mmdet.structures.bbox import bbox_project
from torch.nn import functional as F
import numpy as np
import math
import os.path as osp
import pdb

try:
    import sklearn.mixture as skm
except ImportError:
    skm = None

import numpy as np
from sklearn.ensemble import IsolationForest
from scipy.spatial import distance
from sklearn.preprocessing import MinMaxScaler

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import seaborn as sns

@MODELS.register_module()
class MixPL(SemiBaseDetector):
    """Base class for semi-supervised detectors."""

    def __init__(self,
                 detector: ConfigType,
                 semi_train_cfg: OptConfigType = None,
                 semi_test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            detector=detector,
            semi_train_cfg=semi_train_cfg,
            semi_test_cfg=semi_test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.cache_inputs = []
        self.i = 0

    def loss(self, multi_batch_inputs: Dict[str, Tensor],
             multi_batch_data_samples: Dict[str, SampleList]) -> dict:
        """Calculate losses from multi-branch inputs and data samples.

        Args:
            multi_batch_inputs (Dict[str, Tensor]): The dict of multi-branch
                input images, each value with shape (N, C, H, W).
                Each value should usually be mean centered and std scaled.
            multi_batch_data_samples (Dict[str, List[:obj:`DetDataSample`]]):
                The dict of multi-branch data samples.

        Returns:
            dict: A dictionary of loss components
        """ 
        # pdb.set_trace() 
        losses = dict()
        loss = self.loss_by_gt_instances(multi_batch_inputs['sup'], multi_batch_data_samples['sup'])
        losses.update(loss)  ###监督下的损失

        origin_batch_pseudo_data_samples, batch_info, features = self.get_pseudo_instances(
            multi_batch_inputs['unsup_teacher'], multi_batch_data_samples['unsup_teacher']) #教师模型产生弱增强后的伪标签
        
        self.update_cache(features[0]) #更新缓存

        multi_batch_data_samples['unsup_student'] = self.project_pseudo_instances(
            origin_batch_pseudo_data_samples, multi_batch_data_samples['unsup_student']) # 伪标签投影到学生模型的输入上

        batch_unsup_inputs = copy.deepcopy(multi_batch_inputs['unsup_student'])
        batch_unsup_data_samples = copy.deepcopy(multi_batch_data_samples['unsup_student'])
        # pdb.set_trace()
        batch_unsup_inputs, batch_unsup_data_samples = self.merge(
            *zip(*list(map(self.erase, *self.split(batch_unsup_inputs, batch_unsup_data_samples)))))  ####optical erase


        losses.update(**self.loss_by_pseudo_instances(
                batch_unsup_inputs, batch_unsup_data_samples))

        return losses

    def merge(self, inputs_list, batch_data_samples):
        # pdb.set_trace()
        batch_size = len(inputs_list)
        h, w = 0, 0
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            h, w = max(h, img_h), max(w, img_w)
        h, w = max(h, math.ceil(h / 32) * 32), max(w, math.ceil(w / 32) * 32)
        num_frame = inputs_list[0].shape[0]
        batch_inputs = torch.zeros((batch_size,num_frame, 3, h, w)).to(self.data_preprocessor.device)
        for i in range(batch_size):
            img_h, img_w = batch_data_samples[i].img_shape
            # batch_inputs[i, :, :img_h, :img_w] = inputs_list[i]
            batch_inputs[i,:, :, :img_h, :img_w] = inputs_list[i]
            batch_data_samples[i].set_metainfo({'batch_input_shape': (h, w)})
            batch_data_samples[i].set_metainfo({'pad_shape': (h, w)})
        return batch_inputs, batch_data_samples

    def split(self, batch_inputs, batch_data_samples):
        inputs_list = []
        for i in range(len(batch_inputs)):
            inputs = batch_inputs[i]
            data_samples = batch_data_samples[i]
            img_h, img_w = data_samples.img_shape 
            inputs_list.append(inputs[..., :img_h, :img_w]) ## 将输入按照图像形状进行切片，并添加到输入列表中
            data_samples.pop('batch_input_shape') #
            data_samples.pop('pad_shape')
        return inputs_list, batch_data_samples


    def update_cache(self, batch_inputs: Tensor):
        inputs_list = [batch_inputs[i].unsqueeze(0) for i in range(batch_inputs.size(0))]
        cache_size = self.semi_train_cfg.cache_size
        self.cache_inputs.extend(inputs_list)
        self.cache_inputs = self.cache_inputs[-cache_size:]#如果缓存输入列表的长度超过了缓存大小 cache_size，则对缓存输入列表进行截断，保留最近的 cache_size 个元素
    

    def erase(self, inputs, data_samples):  #inputs: (5, 3, 256, 256)
        inputscopy = inputs.clone()
        inputstemp = inputs.clone()
        inputs = inputs[-1, :, :, :]
        def _get_patches(img_shape): #inputs = inputs[-1, :, :, :]  ##只对关键帧进行erase
            patches = []
            n_patches = np.random.randint(
                self.semi_train_cfg.erase_patches[0], self.semi_train_cfg.erase_patches[1])
            for _ in range(n_patches):
                ratio = np.random.random() * \
                        (self.semi_train_cfg.erase_ratio[1] - self.semi_train_cfg.erase_ratio[0]) + \
                        self.semi_train_cfg.erase_ratio[0]
                ph, pw = int(img_shape[0] * ratio), int(img_shape[1] * ratio)
                px1 = np.random.randint(0, img_shape[1] - pw)
                py1 = np.random.randint(0, img_shape[0] - ph)
                px2, py2 = px1 + pw, py1 + ph
                patches.append([px1, py1, px2, py2])
            return torch.tensor(patches).to(self.data_preprocessor.device)
        erase_patches = _get_patches(data_samples.img_shape)
        for patch in erase_patches:
            px1, py1, px2, py2 = patch
            inputs[:, py1:py2, px1:px2] = 0
        bboxes = data_samples.gt_instances.bboxes
        left_top = torch.maximum(bboxes[:, None, :2], erase_patches[:, :2])
        right_bottom = torch.minimum(bboxes[:, None, 2:], erase_patches[:, 2:])
        wh = torch.clamp(right_bottom - left_top, 0)
        inter_areas = wh[:, :, 0] * wh[:, :, 1]
        bbox_areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        bboxes_erased_ratio = inter_areas.sum(-1) / (bbox_areas + 1e-7)
        valid_inds = bboxes_erased_ratio < self.semi_train_cfg.erase_thr
        data_samples.gt_instances = data_samples.gt_instances[valid_inds]
        inputscopy[-1, :, :, :] = inputs
        assert inputscopy.shape == inputstemp.shape
        return inputscopy, data_samples


    def loss_by_pseudo_instances(self,
                                 batch_inputs: Tensor,
                                 batch_data_samples: SampleList,
                                 batch_info: Optional[dict] = None) -> dict:
        """Calculate losses from a batch of inputs and pseudo data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (List[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`,
                which are `pseudo_instance` or `pseudo_panoptic_seg`
                or `pseudo_sem_seg` in fact.
            batch_info (dict): Batch information of teacher model
                forward propagation process. Defaults to None.

        Returns:
            dict: A dictionary of loss components
        """
        socres_list = []
        boxes_list = []
        for data_samples in batch_data_samples:
           score = data_samples.gt_instances['scores']
           box = data_samples.gt_instances['bboxes']
           boxes_list.append(box)
           socres_list.append(score)
        scores = torch.cat(socres_list, dim=0)
        boxes = torch.cat(boxes_list, dim=0)
        threshold = self.isolationforest_policy(boxes, scores, given_gt_thr=self.semi_train_cfg.cls_pseudo_thr)

        # print('threshold:',threshold)

        batch_data_samples = filter_gt_instances(batch_data_samples, score_thr=threshold)   #score_thr=self.semi_train_cfg.cls_pseudo_thr

        losses,features = self.student.loss(batch_inputs, batch_data_samples)

        # ####可视化features
        # self.i = self.i+1
        # B,C,H,W = features.shape
        # # 1. 将特征展平为二维矩阵 (B, C * H * W)
        # features_flattened = features.view(B*C, -1).detach().cpu().numpy()  # 转为numpy数组以便进行聚类

        # tsne = TSNE()
        # features_flattened = tsne.fit_transform(features_flattened)
        # plt.rcParams ["font.family"] = "serif"
        # plt.rcParams ["font.serif"] = ["Times New Roman"]
        # sns.set(style="white", font_scale=1.0) # white/dark/whitegrid/darkgrid/ticks
        # fig, ax = plt.subplots(figsize=(4.5,4))
        # palette = sns.color_palette("bright", 2)
        # # 生成0到64的整数列表
        # data_list = list(range(B*C))
        # tensor_data = torch.LongTensor(data_list)
        # sns.scatterplot(x=features_flattened[:,0], y=features_flattened[:,1], hue=tensor_data.cpu().numpy(),legend=False, 
        #                 palette=palette, ax=ax, s=3, edgecolors='none', linewidths=0) # 
        # plt.savefig('tsne_'+str(self.i)+'.png')



        ####计算学生模型预测的原型特征和cache中特征的相似度，增强学生模型的学习
        loss = 0
        for fea in self.cache_inputs:
            # loss += F.mse_loss(features, fea.expand(features.shape))
            cosine_similarity = F.cosine_similarity(features, fea.expand(features.shape)).mean()
            loss += 1-cosine_similarity
        
        ###对原型损失超参数消融
        losses['cache_loss'] = loss
        

        pseudo_instances_num = min([len(data_samples.gt_instances) for data_samples in batch_data_samples])
        unsup_weight = self.semi_train_cfg.unsup_weight if pseudo_instances_num >= self.semi_train_cfg.least_num else 0.
        return rename_loss_dict('unsup_', reweight_loss_dict(losses, unsup_weight))
    

    def isolationforest_policy(self,boxes, scores, given_gt_thr=0.4):
        if len(scores) < 4 or len(boxes) < 1:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
            boxes = np.array(boxes)
        
        # 计算每个边界框的中心点坐标
        center_list = []
        for box in boxes:
            box_center = np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
            center_list.append(box_center)
        box_centers = np.array(center_list)
        
        inv_cov_matrix = np.linalg.inv(np.cov(box_centers.T))
        mahalanobis_distances = [distance.mahalanobis(point, np.mean(box_centers, axis=0), inv_cov_matrix) for point in box_centers]
    


        model = IsolationForest(contamination=0.1)  # contamination参数是异常样本比例的估计
        model.fit(box_centers)
        # 计算每个边界框的异常得分
        anomaly_scores = model.decision_function(box_centers)

        # 异常得分归一化
        anomaly_scores_normalized = MinMaxScaler(feature_range=(-1, 1)).fit_transform(anomaly_scores.reshape(-1, 1)).flatten()
        weighted_scores = 0.5 * np.array(mahalanobis_distances) + 0.5 * np.array(anomaly_scores_normalized)
        combined_scores = (weighted_scores - np.min(weighted_scores)) / (np.max(weighted_scores) - np.min(weighted_scores))  # 归一化
        
        threshold = np.percentile(combined_scores, 10)

        if threshold < given_gt_thr:
            threshold = given_gt_thr

        return threshold
    


    def gmm_policy(self, scores, given_gt_thr=0.5, policy='high'):
        """The policy of choosing pseudo label.

        The previous GMM-B policy is used as default.
        1. Use the predicted bbox to fit a GMM with 2 center.
        2. Find the predicted bbox belonging to the positive
            cluster with highest GMM probability.
        3. Take the class score of the finded bbox as gt_thr.

        Args:
            scores (nd.array): The scores.

        Returns:
            float: Found gt_thr.

        """
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        means_init = [[np.min(scores)], [np.max(scores)]]
        weights_init = [1 / 2, 1 / 2]
        precisions_init = [[[1.0]], [[1.0]]]
        gmm = skm.GaussianMixture(
            2,
            weights_init=weights_init,
            means_init=means_init,
            precisions_init=precisions_init)
        gmm.fit(scores)
        gmm_assignment = gmm.predict(scores)
        gmm_scores = gmm.score_samples(scores)
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (
                    scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
                # pos_thr = max(given_gt_thr, pos_thr)
            else:
                pos_thr = given_gt_thr

        return pos_thr
    def bgmm_policy(self,scores, given_gt_thr=0.5, policy='high'):
        if len(scores) < 4:
            return given_gt_thr
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if len(scores.shape) == 1:
            scores = scores[:, np.newaxis]
        bgmm = BayesianGaussianMixture(n_components=2, covariance_type='full',n_init=2)
        bgmm.fit(scores)
        gmm_assignment = bgmm.predict(scores)
        gmm_scores = bgmm.score_samples(scores)
        
        assert policy in ['middle', 'high']
        if policy == 'high':
            if (gmm_assignment == 1).any():
                gmm_scores[gmm_assignment == 0] = -np.inf
                indx = np.argmax(gmm_scores, axis=0)
                pos_indx = (gmm_assignment == 1) & (scores >= scores[indx]).squeeze()
                pos_thr = float(scores[pos_indx].min())
            else:
                pos_thr = given_gt_thr
        elif policy == 'middle':
            if (gmm_assignment == 1).any():
                pos_thr = float(scores[gmm_assignment == 1].min())
            else:
                pos_thr = given_gt_thr

        return pos_thr
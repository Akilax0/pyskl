# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn

from ..builder import RECOGNIZERS
from .base import BaseRecognizer

# importing transformer
# from mmcls.models.backbones import VisionTransformer

import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from PIL import Image

@RECOGNIZERS.register_module()
class PoseViewAttention(BaseRecognizer):
    """3D recognizer model framework."""
    
    def forward_train(self, imgs, label, **kwargs):
        """Defines the computation performed at every call when training."""

        assert self.with_cls_head
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        losses = dict()
        
        print("Heatmaps: ",imgs.shape)
        cpu_tensor = imgs.detach().cpu().numpy()

        # Visualize the Heatmaps using Matplotlib
        plt.imshow(cpu_tensor, cmap='viridis', interpolation='nearest')
        plt.colorbar()
        plt.title("CUDA Float Tensor Visualization")
        plt.savefig("heatmaps.png")  # Save the figure
        plt.close()
        
        x = self.vit(imgs)
        # print("vit output:",x.size(), x.type())

        # cpu_tensor = x.detach().cpu().numpy()

        # # Visualize the tensor using Matplotlib
        # plt.imshow(cpu_tensor, cmap='viridis', interpolation='nearest')
        # plt.colorbar()
        # plt.title("CUDA Float Tensor Visualization")
        # plt.savefig("heatmaps.png")  # Save the figure
        # plt.close()
        
        x = self.extract_feat(x)

        # Visualizing Features 
        # print("SIZES")
        # print("Input Images: ",imgs.size(), imgs.type())
        # print("Outputs size: ",x.size()) 
        
        # matplotlib.image.imsave("input_image.png",imgs[0].cpu())
        # matplotlib.image.imsave("feat.png",out_feat[0][0][0])

        cls_score = self.cls_head(x)
        # print("CLS score size: ",cls_score.size()) 

        gt_label = label.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_label, **kwargs)
        losses.update(loss_cls)

        # print("CLS_SCORE: ",cls_score,gt_label)
        # print("Losses: ",losses)

        return losses

    def forward_test(self, imgs, **kwargs):
        """Defines the computation performed at every call when evaluation, testing."""
        
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
       
        imgs = self.vit(imgs)

        if self.max_testing_views is not None:
            total_views = imgs.shape[0]
            assert num_segs == total_views, (
                'max_testing_views is only compatible '
                'with batch_size == 1')
            view_ptr = 0
            feats = []
            while view_ptr < total_views:
                batch_imgs = imgs[view_ptr:view_ptr + self.max_testing_views]
                
                # vit
                x = self.extract_feat(batch_imgs)
                feats.append(x)
                view_ptr += self.max_testing_views
            # should consider the case that feat is a tuple
            if isinstance(feats[0], tuple):
                len_tuple = len(feats[0])
                feat = [
                    torch.cat([x[i] for x in feats]) for i in range(len_tuple)
                ]
                feat = tuple(feat)
            else:
                feat = torch.cat(feats)
        else:
            # vit
            feat = self.extract_feat(imgs)

        if self.test_cfg.get('feat_ext', False):
            feat_dim = len(feat[0].size()) if isinstance(feat, tuple) else len(feat.size())
            assert feat_dim in [5, 2], (
                'Got feature of unknown architecture, '
                'only 3D-CNN-like ([N, in_channels, T, H, W]), and '
                'transformer-like ([N, in_channels]) features are supported.')
            if feat_dim == 5:  # 3D-CNN architecture
                # perform spatio-temporal pooling
                avg_pool = nn.AdaptiveAvgPool3d(1)
                if isinstance(feat, tuple):
                    feat = [avg_pool(x) for x in feat]
                    # concat them
                    feat = torch.cat(feat, axis=1)
                else:
                    feat = avg_pool(feat)
                # squeeze dimensions
                feat = feat.reshape((batches, num_segs, -1))
                # temporal average pooling
                feat = feat.mean(axis=1)
            return feat.cpu().numpy()

        # should have cls_head if not extracting features
        assert self.with_cls_head
        cls_score = self.cls_head(feat)
        cls_score = cls_score.reshape(batches, num_segs, cls_score.shape[-1])
        cls_score = self.average_clip(cls_score)
        return cls_score.cpu().numpy()

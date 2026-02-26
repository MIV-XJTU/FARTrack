"""
Basic OSTrack model.
"""
from copy import deepcopy
import math
import os
from typing import List
import re

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.fartrack_sparse.vit import vit_base_patch16_224, vit_large_patch16_224, vit_tiny_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.layers.mask_decoder import build_maskdecoder

from lib.models.layers.head import build_decoder, MLP, DropPathAllocator

import time


class FARTrackSparse(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer,
                 ):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
 
        # tiny model
        self.identity = torch.nn.Parameter(torch.zeros(1, 6, 192)) # 6: 5 templates + 1 search

        self.identity = trunc_normal_(self.identity, std=.02)


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                seq_input=None,
                head_type=None,
                stage=None,
                search_feature=None,
                target_in_search_img=None,
                gt_bboxes=None,
                mask=None,
                ):
        template_0 = template
        out, z_0_feat, z_1_feat, x_feat, score_feat = self.backbone(z_0=template_0, x=search, identity=self.identity, seqs_input=seq_input, mask=mask, stage=stage,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,)

        return out
        


def build_fartrack_sparse(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = "/home/caoanjia/wgj/FARTrack-main/pretrained_models/" # Set the path to your pretrained_models.
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        print("i use vit_large")
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224':
        print("i use vit_tiny")
        backbone = vit_tiny_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION, prenum=cfg.MODEL.PRENUM)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)


    model = FARTrackSparse(
        backbone,
    )
    load_from = cfg.MODEL.PRETRAIN_PTH
    checkpoint = torch.load(load_from, map_location="cpu")
    new_checkpoints = checkpoint['net'].copy()
    for key in checkpoint['net'].keys():
        pattern = re.compile(r'(\w+\.)*(norm\d+)(\.\w+)*')
        if 'norm' in key and 'masknorm' not in key and 'norm.' not in key:
            match = pattern.match(key)
            if match:
                norm_part = match.group(2)
                masknorm_key = key.replace(norm_part, f'mask{norm_part}.norm')
          
            new_checkpoints[masknorm_key] = checkpoint['net'][key]
    
    checkpoint["net"] = new_checkpoints
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    print('Load pretrained model from: ' + load_from)
    print(missing_keys)
    print(unexpected_keys)
    if 'sequence' in cfg.MODEL.PRETRAIN_FILE and training:
        print("i change myself")
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model

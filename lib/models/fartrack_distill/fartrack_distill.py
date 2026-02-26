from copy import deepcopy
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.fartrack_distill.vit import vit_base_patch16_224, vit_large_patch16_224, vit_tiny_patch16_224
from lib.utils.box_ops import box_xyxy_to_cxcywh


class FARTrackDistill(nn.Module):

    def __init__(self,
                 transformer,
                 ):

        super().__init__()
        # tiny model
        self.identity = torch.nn.Parameter(torch.zeros(1, 6, 192))
        
        self.identity = trunc_normal_(self.identity, std=.02)        
        
        self.backbone = transformer

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                seq_input=None,
                target_in_search_img=None,
                gt_bboxes=None,
                ):
        
        template_0 = template
        
       # template_1 = template[1]
        out, z_0_feat, z_1_feat, x_feat = self.backbone(z_0=template_0, z_1=None, x=search, identity=self.identity, seqs_input=seq_input,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn,)

        return out



def build_fartrack_distill(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('ARTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    print("pretrained file: ", pretrained)
    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        print("i use vit_large")
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224':
        print("i use vit_tiny")
        backbone = vit_tiny_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, bins=cfg.MODEL.BINS, range=cfg.MODEL.RANGE, extension=cfg.MODEL.EXTENSION)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)


    model = FARTrackDistill(
        backbone,
    )

    # load_from = "/data5/FARTrack-re/main/fartrack_distill/FARTrack_ep0500.pth.tar" # AO: 0.673
    load_from = cfg.MODEL.PRETRAIN_PTH
    checkpoint = torch.load(load_from, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
    print('Load pretrained model from: ' + load_from)
    
    return model

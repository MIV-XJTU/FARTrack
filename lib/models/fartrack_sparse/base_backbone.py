# -*- coding: utf-8 -*-
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import resize_pos_embed
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.fartrack_sparse.utils import combine_tokens, recover_tokens


import time

def generate_square_subsequent_mask(sz, sx, ss):
    r"""Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
    """
    # 0 means mask, 1 means visible
    sum = sz + sx + ss
    mask = (torch.triu(torch.ones(sum, sum)) == 1).transpose(0, 1)
    mask[:, :] = 0
    mask[:int(sz), :int(sz)] = 1
  #  mask[:int(sz/2), :int(sz/2)] = 1 #template self
   # mask[int(sz/2):sz, int(sz/2):sz] = 1 # dt self
   # mask[int(sz/2):sz, sz:sz+sx] = 1 # dt search
#    mask[int(sz / 2):sz, -1] = 1  # dt search
    mask[sz:sz+sx, :sz+sx] = 1 # sr dt-t-sr
    mask[sz+sx:, :] = 1 # co dt-t-sr-co
    return ~mask

class BaseBackbone(nn.Module):
    def __init__(self):
        super().__init__()

        # for original ViT
        self.pos_embed = None
        self.img_size = [224, 224]
        self.patch_size = 16
        self.embed_dim = 384

        self.cat_mode = 'direct'

        self.pos_embed_z = None
        self.pos_embed_x = None

        self.template_segment_pos_embed = None
        self.search_segment_pos_embed = None

        self.return_inter = False
        self.return_stage = [2, 5, 8, 11]

        self.add_cls_token = False
        self.add_sep_seg = False
        self.random_z = None

    def finetune_track(self, cfg, patch_start_index=1):

        search_size = to_2tuple(cfg.DATA.SEARCH.SIZE)
        template_size = to_2tuple(cfg.DATA.TEMPLATE.SIZE)
        new_patch_size = cfg.MODEL.BACKBONE.STRIDE

        self.cat_mode = cfg.MODEL.BACKBONE.CAT_MODE
        self.return_inter = cfg.MODEL.RETURN_INTER
        self.add_sep_seg = cfg.MODEL.BACKBONE.SEP_SEG

        # resize patch embedding
        if new_patch_size != self.patch_size:
            print('Inconsistent Patch Size With The Pretrained Weights, Interpolate The Weight!')
            old_patch_embed = {}
            for name, param in self.patch_embed.named_parameters():
                if 'weight' in name:
                    param = nn.functional.interpolate(param, size=(new_patch_size, new_patch_size),
                                                      mode='bicubic', align_corners=False)
                    param = nn.Parameter(param)
                old_patch_embed[name] = param
            self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=new_patch_size, in_chans=3,
                                          embed_dim=self.embed_dim)
            self.patch_embed.proj.bias = old_patch_embed['proj.bias']
            self.patch_embed.proj.weight = old_patch_embed['proj.weight']

        # for patch embedding
        patch_pos_embed = self.pos_embed[:, patch_start_index:, :]
        patch_pos_embed = patch_pos_embed.transpose(1, 2)
        B, E, Q = patch_pos_embed.shape
        P_H, P_W = self.img_size[0] // self.patch_size, self.img_size[1] // self.patch_size
        patch_pos_embed = patch_pos_embed.view(B, E, P_H, P_W)

        # for search region
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        search_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                           align_corners=False)
        search_patch_pos_embed = search_patch_pos_embed.flatten(2).transpose(1, 2)

        # for template region
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        template_patch_pos_embed = nn.functional.interpolate(patch_pos_embed, size=(new_P_H, new_P_W), mode='bicubic',
                                                             align_corners=False)
        template_patch_pos_embed = template_patch_pos_embed.flatten(2).transpose(1, 2)

        self.pos_embed_z = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_z0 = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_z1 = nn.Parameter(template_patch_pos_embed)
        self.pos_embed_x = nn.Parameter(search_patch_pos_embed)

        # for cls token (keep it but not used)
        if self.add_cls_token and patch_start_index > 0:
            cls_pos_embed = self.pos_embed[:, 0:1, :]
            self.cls_pos_embed = nn.Parameter(cls_pos_embed)

        # separate token and segment token
        if self.add_sep_seg:
            self.template_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.template_segment_pos_embed = trunc_normal_(self.template_segment_pos_embed, std=.02)
            self.search_segment_pos_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            self.search_segment_pos_embed = trunc_normal_(self.search_segment_pos_embed, std=.02)

        # self.cls_token = None
        # self.pos_embed = None

        if self.return_inter:
            for i_layer in self.fpn_stage:
                if i_layer != 11:
                    norm_layer = partial(nn.LayerNorm, eps=1e-6)
                    layer = norm_layer(self.embed_dim)
                    layer_name = f'norm{i_layer}'
                    self.add_module(layer_name, layer)
    
    def mask_generation(self, attn_avg, inference=False, template_token_num=None):
    
        template_pruning = 5
        
        H_z = 112
        H = 224
        B = attn_avg.shape[0]
        
        template_patch_shape = int((H_z / 16) ** 2)
        search_patch_shape = int((H / 16) ** 2)
        
        if inference:
            attn_coor = attn_avg[:, -4:, :template_token_num].mean(dim=-2)
        
            center_size = 1
            search_tokens = attn_avg[:, -200:-4,  :template_token_num] 
            search_tokens = search_tokens.view(B, 14, 14, template_token_num) 

            center_y = H //16 // 2
            center_x = H // 16 // 2

            half_size = center_size // 2
            start_y = max(0, center_y - half_size)
            end_y = min(14, center_y + half_size + 1)
            start_x = max(0, center_x - half_size)
            end_x = min(14, center_x + half_size + 1)

            center_attn = search_tokens[:, start_y:end_y, start_x:end_x]
            attn_search = center_attn.reshape(B, center_size**2, template_token_num).mean(dim=-2)
            # print(attn_coor.shape)
            # print(attn_search.shape)
                
            # attn_mix = attn_coor + attn_search
            attn_mix = attn_search# + attn_coor

            attn_mix = attn_mix.reshape(B, template_token_num)
            
            def inference_prune_attention_map(attention_map):
                final_mask = []
                B, num_template, template_tokens = attention_map.shape
                sorted_indices = torch.argsort(attention_map, dim=-1)
                prune_num = int(template_tokens * 0.25)
                mask = torch.ones_like(attention_map, dtype=torch.bool)
            
                for b in range(B):
                    for t in range(num_template):
                        mask[b, t, sorted_indices[b, t, :prune_num]] = 0

                flattened_mask = mask.view(B, -1)
                final_mask.append(flattened_mask)

                prune_num = int(template_tokens * 0.5)
                mask = torch.ones_like(attention_map, dtype=torch.bool)

                for b in range(B):
                    for t in range(num_template):
                        mask[b, t, sorted_indices[b, t, :prune_num]] = 0

                flattened_mask = mask.view(B, -1)
                final_mask.append(flattened_mask)

                prune_num = int(template_tokens * 0.75)
                mask = torch.ones_like(attention_map, dtype=torch.bool)

                for b in range(B):
                    for t in range(num_template):
                        mask[b, t, sorted_indices[b, t, :prune_num]] = 0

                flattened_mask = mask.view(B, -1)
                final_mask.append(flattened_mask)

                prune_num = int(template_tokens * 0.9)
                mask = torch.ones_like(attention_map, dtype=torch.bool)

                for b in range(B):
                    for t in range(num_template):
                        mask[b, t, sorted_indices[b, t, :prune_num]] = 0

                flattened_mask = mask.view(B, -1)
                final_mask.append(flattened_mask)

            
                return final_mask
            mask = inference_prune_attention_map(attn_mix[:, -49:].unsqueeze(1))

            return mask
        	            
        attn_coor = attn_avg[:, -4:, :template_patch_shape * template_pruning].mean(dim=-2)
        
        center_size = 1
        search_tokens = attn_avg[:, -200:-4,  :template_patch_shape * template_pruning]  
        search_tokens = search_tokens.view(B, 14, 14, template_patch_shape * template_pruning)  

        center_y = H //16 // 2
        center_x = H // 16 // 2

        half_size = center_size // 2
        start_y = max(0, center_y - half_size)
        end_y = min(14, center_y + half_size + 1)
        start_x = max(0, center_x - half_size)
        end_x = min(14, center_x + half_size + 1)
      #  print(start_y, end_y, start_x, end_x)

        center_attn = search_tokens[:, start_y:end_y, start_x:end_x]
        attn_search = center_attn.reshape(B, center_size**2, template_patch_shape * template_pruning).mean(dim=-2)
       # attn_mix = attn_coor + attn_search
        attn_mix =  attn_search

        attn_mix = attn_mix.reshape(B, template_pruning, template_patch_shape)
       # print(attn_mix[0])
        
        def prune_attention_map(attention_map, extra_tokens):
            B, num_template, template_tokens = attention_map.shape
            sorted_indices = torch.argsort(attention_map, dim=-1)
            prune_num = int(template_tokens * 0.25)
            mask = torch.ones_like(attention_map, dtype=torch.bool)
            
            for b in range(B):
                for t in range(num_template-1):
                    mask[b, t, sorted_indices[b, t, :prune_num]] = 0
            
            flattened_mask = mask.view(B, -1)
            final_mask = torch.cat([flattened_mask, torch.ones(B, extra_tokens, dtype=torch.bool, device=attention_map.device)], dim=-1)
            
            return final_mask
        mask = prune_attention_map(attn_mix, 200)
        N = mask.shape[1]
        new_mask = mask.unsqueeze(-1).expand(-1, -1, N)
        new_mask = new_mask.permute(0, 2, 1)
     
        return new_mask
    
    def forward_features(self, z_0, x, identity, seqs_input, mask, stage):

        mask = mask.to(x.device)
        share_weight = self.word_embeddings.weight.T
        out_list = []

        x0 = self.bins * self.range
        y0 = self.bins * self.range + 1
        x1 = self.bins * self.range + 2
        y1 = self.bins * self.range + 3

        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        command = torch.cat([torch.ones((B, 1)).to(x) * x0, torch.ones((B, 1)).to(x) * y0,
                      torch.ones((B, 1)).to(x) * x1,
                      torch.ones((B, 1)).to(x) * y1], dim=1)
        trajectory = seqs_input
        command = command.to(trajectory)
        seqs_input_ = command
        
        seqs_input_ = seqs_input_.to(torch.int64).to(x.device)

        tgt = self.word_embeddings(seqs_input_).permute(1, 0, 2)
        
        x = self.patch_embed(x)
        x_feat = x.clone()

        try:
            z_0 = torch.stack(z_0, dim=1)
        except:
            z_0 = z_0
        B, L, _, H_z = z_0.shape[:4]
        

        z_0 = z_0.reshape(B*L, -1, *z_0.shape[3:])  # [B*L, C, H, W]
        
# patch embedding
        z_ = self.patch_embed(z_0)  # [B*L, P, D] where P is patches，D is embedding dim

        pos_embed = self.pos_embed_z0.expand(L, -1, -1)  # [L, P, D]
        pos_embed = pos_embed.repeat(B, 1, 1)  # [B*L, P, D]

        identity_ = identity[:, :-1, :].unsqueeze(2).expand(B, -1, z_.shape[1], -1)  # [B, L, P, D]
        identity_ = identity_.reshape(B*L, -1, identity_.shape[-1])  # [B*L, P, D]

        z_ = z_ + pos_embed + identity_

        z_ = z_.reshape(B, L, -1, z_.shape[-1])  # [B, L, P, D]
        z_ = z_.reshape(B, L*z_.shape[2], -1)  # [B, L*P, D]
        template_token_num = z_.shape[1]


        x += identity[:, -1, :].repeat(B, self.pos_embed_x.shape[1], 1)

        query_command_embed_ = self.position_embeddings.weight.unsqueeze(1)
        prev_embed_ = self.prev_position_embeddings.weight.unsqueeze(1)

        query_seq_embed = torch.cat([query_command_embed_], dim=0)

        query_seq_embed = query_seq_embed.repeat(1, B, 1)

        tgt = tgt.transpose(0, 1)
        query_seq_embed = query_seq_embed.transpose(0, 1)

        x += self.pos_embed_x


        tgt += query_seq_embed[:, :]

        z = z_

        zx = torch.cat((z, x), dim=1)
        zxs = torch.cat((zx, tgt), dim=1)

        zxs = self.pos_drop(zxs)
        attn_sum = None
        #print(mask[0][0])
        for j, blk in enumerate(self.blocks): # 0~11
            # if j == 10:
            #     break
            zxs, attn = blk(zxs, return_attention=True, padding_mask=mask)
            attn = attn.detach()
            if attn_sum == None:
                attn_sum = attn
            else:
                attn_sum = attn_sum + attn
        for j, blk in enumerate(self.extension): # 0~2
            # if j == 0:
            #     break
            zxs, attn = blk(zxs, return_attention=True, padding_mask=mask)
            attn = attn.detach()
            attn_sum = attn_sum + attn

        attn_avg = attn_sum.mean(dim=1)

        if stage == "inference":
            mask_ = self.mask_generation(attn_avg, inference=True, template_token_num=template_token_num)
        else:
            if torch.all(mask).item():
                mask_ = self.mask_generation(attn_avg, False)
            else:
                mask_ = mask

        x_out = self.norm(zxs[:, -4:])
        
        seq_feat = x_out

        possibility = torch.matmul(x_out, share_weight)
        out = possibility + self.output_bias
        temp = out.transpose(0, 1)

        out_list.append(out.unsqueeze(0))
        out = out.softmax(-1)

        value, extra_seq = out.topk(dim=-1, k=1)[0], out.topk(dim=-1, k=1)[1]
        for i in range(4):
            value, extra_seq = out[:, i, :].topk(dim=-1, k=1)[0], out[:, i, :].topk(dim=-1, k=1)[1]
            if i == 0:
                seqs_output = extra_seq
                values = value
            else:
                seqs_output = torch.cat([seqs_output, extra_seq], dim=-1)
                values = torch.cat([values, value], dim=-1)

        output = {'seqs': seqs_output, 'class': values, 'feat': temp, "state": "val/test", "seq_feat": seq_feat.permute(1, 0, 2), "x_feat": x_feat, "mask":mask_}

        return output, None, None, None, None

    def forward(self, z_0, x, identity, seqs_input, mask, stage, **kwargs):
        """
        Joint feature extraction and relation modeling for the basic ViT backbone.
        Args:
            z (torch.Tensor): template feature, [B, C, H_z, W_z]
            x (torch.Tensor): search region feature, [B, C, H_x, W_x]

        Returns:
            x (torch.Tensor): merged template and search region feature, [B, L_z+L_x, C]
            attn : None
        """
        output = self.forward_features(z_0, x, identity, seqs_input, mask, stage)

        return output

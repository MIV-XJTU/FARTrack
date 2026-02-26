import math

from lib.models.fartrack_sparse import build_fartrack_sparse
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target, transform_image_to_crop
# for debug
import cv2
import os
import numpy as np
import shutil
from PIL import Image
import random

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import re

std = np.array([0.229, 0.224, 0.225])
mean = np.array([0.485, 0.456, 0.406])
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def apply_mask_and_generate_mask_image(image_path, mask_tensor, output_image_path, output_mask_path=None):
    overlay_alpha = 128
    img = Image.open(image_path).convert("RGBA")
    width, height = img.size
    data = np.array(img)

    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    overlay_data = np.array(overlay)

    mask_np = mask_tensor.cpu().numpy().reshape(7, 7)

    patch_size = 16
    image_size = 112

    patch_size = 16
    for i in range(7):
        for j in range(7):
            if mask_np[i, j] == 0:
                y_start = i * patch_size
                y_end = (i + 1) * patch_size
                x_start = j * patch_size
                x_end = (j + 1) * patch_size

                overlay_data[y_start:y_end, x_start:x_end] = [255, 255, 255, overlay_alpha]

    overlay = Image.fromarray(overlay_data, 'RGBA')
    composite = Image.alpha_composite(img, overlay)

    overlay = Image.fromarray(overlay_data, 'RGBA')
    composite.save(output_image_path)

    # mask_array = np.zeros((image_size, image_size), dtype=np.uint8)
    # for i in range(7):
    #     for j in range(7):
    #         if mask_np[i, j] == 0:
    #             mask_array[
    #             i * patch_size: (i + 1) * patch_size,
    #             j * patch_size: (j + 1) * patch_size
    #             ] = 255
    # mask_img = Image.fromarray(mask_array, mode='L')
    # mask_img.save(output_mask_path)

class FARTrackSparse(BaseTracker):
    def __init__(self, params, dataset_name):
        super(FARTrackSparse, self).__init__(params)
        network = build_fartrack_sparse(params.cfg, training=False) # The checkpoint parameters are loaded in build_network
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True) # default strict=True
        print("Load pretrained FARTrack from: ", self.params.checkpoint)
        self.cfg = params.cfg
        self.bins = params.cfg.MODEL.BINS
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.dz_feat = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.num_template = self.cfg.DATA.TEMPLATE.NUMBER # 5
        self.use_visdom = params.debug
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}
        self.store_result = None
        self.prenum = params.cfg.MODEL.PRENUM
        self.range = params.cfg.MODEL.RANGE
        self.x_feat = None
        self.allnum = 0


    def template_update_sampling(self, new_z, sampling_method="linear", mask=None):
        #  """
        #   Update the template to support multiple sampling methods.
            
        #   Args:
        #     new_z: New template.
        #     sampling_method: Sampling method, supporting the following options:
        #       - "linear": Linear uniform sampling. # 95 762
        #       - "exponential": Exponential decay sampling (denser for recent frames). 0.5 75.3 0.25 74.8 0.75 77 0.80 76.8 0.7 77.1
        #       - "logarithmic": Logarithmic increasing sampling (denser for earlier frames). 76.1
        #       - "random": Random sampling (the 0th frame and the latest frame are always retained).
        #       - "fixed_weight": Fixed interval decreasing weight sampling.
        #  """
        if not hasattr(self, 'stored_templates'):
            self.stored_templates = [] 
            self.stored_templates.append(self.z_dict1[0])  
            self.store_mask = [] # mask 25%
            self.store_mask_50 = [] # mask 50%
            self.store_mask_75 = []
            self.store_mask_90 = []
          #  self.store_mask.append(mask[:, 49*4:49*5])


        if self.frame_id >= 1:
            self.stored_templates.append(new_z)

            self.store_mask.append(mask[0]) # (mask[0])
            self.store_mask_50.append(mask[1])
            self.store_mask_75.append(mask[2])
            self.store_mask_90.append(mask[3])


        current_frame_count = self.frame_id + 1 
        num_templates = self.num_template  
        mask_temp = torch.ones(1, 445) > 0

        if current_frame_count < num_templates:
            mode = 'mode1'  
    
            for template_pos in range(num_templates):
        # mode1：to（0,0,0,1,2）
                if mode == 'mode1':
                    shift = num_templates - current_frame_count
                    template_idx = max(0, template_pos - shift)
        
        # mode2：to（0,1,1,1,1）
                elif mode == 'mode2':
                    template_idx = min(template_pos, current_frame_count - 1)
        
                template_idx = min(template_idx, len(self.stored_templates)-1)
        
                self.z_dict1[template_pos] = self.stored_templates[template_idx]
                if template_idx < len(self.store_mask):
                  #  print(template_idx, len(self.store_mask))
                    mask_temp[:, 49*template_pos:49*(template_pos+1)] = self.store_mask[template_idx]
                    #print(mask_temp)
                else:
                    mask_extra = torch.ones([1, 49]) > 0
                    mask_temp[:, 49*template_pos:49*(template_pos+1)] = mask_extra
               
                #print(template_pos, template_idx)
            mask_temp = mask_temp.unsqueeze(-1).expand(-1, -1, 445).permute(0, 2, 1)
            #print(mask_temp)
            self.mask = mask_temp
            return
 
        if sampling_method == "linear":

            step = (current_frame_count - 1) / (num_templates - 1)
            sampled_indices = [int(i * step) for i in range(num_templates - 1)]
            sampled_indices.append(current_frame_count - 1)

        elif sampling_method == "exponential":
 
            sampled_indices = [0] 
            for i in range(1, num_templates - 1):
                index = int((current_frame_count - 1) * (1 - 0.7 ** i))
                sampled_indices.append(index)
            sampled_indices.append(current_frame_count - 1) 

        elif sampling_method == "logarithmic":
            sampled_indices = [0] 
            for i in range(1, num_templates - 1):
                index = int((current_frame_count - 1) * (i / (num_templates - 1)) ** 0.5)
                sampled_indices.append(index)
            sampled_indices.append(current_frame_count - 1) 

        elif sampling_method == "random":
            sampled_indices = [0, current_frame_count - 1]  
            if num_templates > 2:
                additional_indices = sorted(
                    random.sample(range(1, current_frame_count - 1), num_templates - 2)
                )
                sampled_indices = [0] + additional_indices + [current_frame_count - 1]

        elif sampling_method == "fixed_weight":
            weights = [(1 / (i + 1)) for i in range(current_frame_count)]
            total_weight = sum(weights)
            probabilities = [w / total_weight for w in weights]
            sampled_indices = sorted(
                random.choices(range(current_frame_count), weights=probabilities, k=num_templates - 1)
            )
            sampled_indices[0] = 0  
            sampled_indices[-1] = current_frame_count - 1 

        else:
            raise ValueError(f"Unknown sampling method: {sampling_method}")


        for i, idx in enumerate(sampled_indices):
            self.z_dict1[i] = self.stored_templates[idx]
            if idx < len(self.store_mask):
                mask_temp[:, 49*i:49*(i+1)] = self.store_mask[idx]
            else:
                mask_extra = torch.ones([1, 49]) > 0
                mask_temp[:, 49*i:49*(i+1)] = mask_extra
        mask_temp = mask_temp.unsqueeze(-1).expand(-1, -1, 445).permute(0, 2, 1)
        self.mask = mask_temp

    def initialize(self, image, info: dict, name:str):
        # forward the template once
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        
        self.root = os.path.join(project_root, "got10k_template_mask")
        # self.root = f"/data5/FARTrack/main/got10k_template_mask/"
        self.name = name
        self.x_feat = None
        self.update_ = False

        try: 
            os.makedirs(self.root, exist_ok=True)
        except OSError as e:
            print(f"mkdir failed: {e}")

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                                output_sz=self.params.template_size)  # output_sz=self.params.template_size
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            # initialize dynamic template as template in first frame
            self.z_dict1 = [template.tensors] * self.num_template

        self.box_mask_z = None
        self.mask = torch.ones([1, 445, 445]) > 0
        self.mask = self.mask.cuda()

        # save states
        self.state = info['init_bbox']
        self.store_result = [info['init_bbox'].copy()]
        for i in range(self.prenum - 1): # prenum：3
            self.store_result.append(info['init_bbox'].copy())
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        path = self.root + self.name
        # if self.frame_id==0:
        #     self.allnum+=1
        #     path = self.root + self.name
        #     if os.path.exists(path):
        #         shutil.rmtree(path)
        #     os.mkdir(path)
        #     os.mkdir(path+"/template_whole/")


        H, W, _ = image.shape
        self.frame_id += 1

        # if self.frame_id == 1:
        #     name = "template_0" + ".png"
        #     template = self.z_dict1[0].cpu().reshape(3, 112, 112).permute(1, 2, 0).numpy()
        #     template = np.clip(template * std + mean, 0, 1) * 255
        #     im = Image.fromarray(np.uint8((template)))
        #     im.save(path + "/" + name)
        # else:
        #     name = "template_" + str(self.frame_id - 1) + ".png"
        #     template = self.z_dict1[-1].cpu().reshape(3, 112, 112).permute(1, 2, 0).numpy()
        #     template = np.clip(template * std + mean, 0, 1) * 255
        #     im = Image.fromarray(np.uint8((template)))
        #     im.save(path + "/" + name)

        #     idx = self.frame_id - 2
        #     before_name = "template_" + str(self.frame_id - 2) + ".png"
        #     before_path = path + "/" + before_name
        #     after_name = "template_mask_25_" + str(self.frame_id - 2) + ".png"
        #     after_path = path + "/template_whole/" + after_name
        #     before_mask = self.store_mask[idx]
        #     apply_mask_and_generate_mask_image(image_path=before_path,
        #                                        mask_tensor=before_mask,
        #                                        output_image_path=after_path)
        #     after_name = "template_mask_50_" + str(self.frame_id - 2) + ".png"
        #     after_path = path + "/template_whole/" + after_name
        #     before_mask = self.store_mask_50[idx]
        #     apply_mask_and_generate_mask_image(image_path=before_path,
        #                                        mask_tensor=before_mask,
        #                                        output_image_path=after_path)
        #     after_name = "template_mask_75_" + str(self.frame_id - 2) + ".png"
        #     after_path = path + "/template_whole/" + after_name
        #     before_mask = self.store_mask_75[idx]
        #     apply_mask_and_generate_mask_image(image_path=before_path,
        #                                        mask_tensor=before_mask,
        #                                        output_image_path=after_path)
        #     after_name = "template_mask_90_" + str(self.frame_id - 2) + ".png"
        #     after_path = path + "/template_whole/" + after_name
        #     before_mask = self.store_mask_90[idx]
        #     apply_mask_and_generate_mask_image(image_path=before_path,
        #                                        mask_tensor=before_mask,
        #                                        output_image_path=after_path)



        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)

        for i in range(len(self.store_result)):
            box_temp = self.store_result[i].copy()
            box_out_i = transform_image_to_crop(torch.Tensor(self.store_result[i]), torch.Tensor(self.state),
                                                resize_factor,
                                                torch.Tensor([self.cfg.TEST.SEARCH_SIZE, self.cfg.TEST.SEARCH_SIZE]),
                                                normalize=True)
            box_out_i[2] = box_out_i[2] + box_out_i[0]
            box_out_i[3] = box_out_i[3] + box_out_i[1]
            box_out_i = box_out_i.clamp(min=-0.5, max=1.5)
            box_out_i = (box_out_i + 0.5) * (self.bins - 1)
            if i == 0:
                seqs_out = box_out_i
            else:
                seqs_out = torch.cat((seqs_out, box_out_i), dim=-1)

        seqs_out = seqs_out.unsqueeze(0)

        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search

            out_dict = self.network.forward(
                template=self.z_dict1, search=x_dict.tensors, ce_template_mask=self.box_mask_z,
                seq_input=seqs_out, stage="inference", search_feature=None, mask=self.mask)


        mask = out_dict['mask']

        pred_boxes = (out_dict['seqs'][:, 0:4] + 0.5) / (self.bins - 1) - 0.5

        pred_feat = out_dict['feat']
        pred = pred_feat.permute(1, 0, 2).reshape(-1, self.bins * self.range + 5)

        pred = pred_feat[0:4, :, 0:self.bins * self.range]

        out = pred.softmax(-1).to(pred)
        #mul = torch.range((-1 * self.range * 0.5 + 0.5) + 1 / (self.bins * self.range), (self.range * 0.5 + 0.5) - 1 / (self.bins * self.range), 2 / (self.bins * self.range)).to(pred)
        mul = torch.range((-1 * self.range * 0.5 + 0.5), (self.range * 0.5 + 0.5) - 1 / (self.bins * self.range), 2 / (self.bins * self.range)).to(pred)

        ans = out * mul
        ans = ans.sum(dim=-1)
        ans = ans.permute(1, 0).to(pred)
        
        #pred_boxes = ans

        pred_boxes = (ans + pred_boxes) / 2

        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)

        pred_new = pred_boxes
        pred_new[2] = pred_boxes[2] - pred_boxes[0]
        pred_new[3] = pred_boxes[3] - pred_boxes[1]
        pred_new[0] = pred_boxes[0] + pred_new[2] / 2
        pred_new[1] = pred_boxes[1] + pred_new[3] / 2
 

        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()

        self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)

        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                output_sz=self.params.template_size)  # (x1, y1, w, h)
        new_z = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors   

        self.template_update_sampling(new_z, "exponential", mask=mask) # exponential

        # Used to store trajectory information for the first prenum frames/trajectories.
        if len(self.store_result) < self.prenum: # prenum：3
            self.store_result.append(self.state.copy()) # Two init_bbox values are stored in advance.
        else:
            for i in range(self.prenum): # i: 0, 1, 2
                if i != self.prenum - 1:
                    self.store_result[i] = self.store_result[i + 1]
                else:
                    self.store_result[i] = self.state.copy() 

        # for debug
        if self.debug:
            if not self.use_visdom:
                x1, y1, w, h = self.state
                image_BGR = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
                save_path = os.path.join(self.save_dir, "%04d.jpg" % self.frame_id)
                cv2.imwrite(save_path, image_BGR)
            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap',
                                     1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        # cx_real = cx + cx_prev
        # cy_real = cy + cy_prev
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1)  # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return FARTrackSparse

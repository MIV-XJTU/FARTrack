import math

from lib.models.fartrack_distill import build_fartrack_distill
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
# for debug
import cv2
import os
import numpy as np
import shutil
from PIL import Image, ImageDraw, ImageFont

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond
import random

from visualize import visualize_region_attention, visualize_grid_attention, visualize_grid_attention_v2
from visualize import draw_line_chart

std = np.array([0.229, 0.224, 0.225])
mean = np.array([0.485, 0.456, 0.406])
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def concatenate_attention_maps(original_image_path, template_image_path, layer_image_paths, output_path,
                               target_height=None):
    """
    Overlay the template on the top-left corner of the original search image, and concatenate it with multi-layer attention maps.

    Args:
        original_image_path (str): Path to the original search image
        template_image_path (str): Path to the template image
        layer_image_paths (list): List of paths to attention maps of each layer
        output_path (str): Path to save the concatenated image
        target_height (int, optional): Unified display height
    """

    search_img = Image.open(original_image_path).convert('RGB')
    template_img = Image.open(template_image_path).convert('RGB')
    search_img.paste(template_img, (0, 0))  

    layer_images = [Image.open(path) for path in layer_image_paths]

    spacing = 20  

    if target_height is None:
        target_height = search_img.height

    def resize_image(img):
        if img.height == target_height:
            return img
        ratio = target_height / img.height
        new_width = int(img.width * ratio)
        return img.resize((new_width, target_height), Image.ANTIALIAS)

    search_img = resize_image(search_img)
    layer_images = [resize_image(img) for img in layer_images]

    total_width = search_img.width + sum(img.width for img in layer_images) + spacing * len(layer_images)

    concatenated_img = Image.new('RGB', (total_width, target_height), (255, 255, 255))

    x_offset = 0
    all_images = [search_img] + layer_images

    for idx, img in enumerate(all_images):
        concatenated_img.paste(img, (x_offset, 0))
        if idx != len(all_images) - 1:
            x_offset += img.width + spacing

    concatenated_img.save(output_path)
    print(f"The concatenated result has been saved to：{output_path}")

def run_grid_attention_example(img_path="visualize/test_data/example.jpg", save_path="test_grid_attention/", attention_mask=None, version=2, quality=100):

    normed_attention_mask = attention_mask.reshape(int(attention_mask.shape[-1]**0.5), int(attention_mask.shape[-1]**0.5))

    assert version in [1, 2], "We only support two version of attention visualization example"
    if version == 1:
        visualize_grid_attention(img_path=img_path,
                                save_path=save_path,
                                attention_mask=normed_attention_mask,
                                save_image=True,
                                save_original_image=True,
                                quality=quality)
    elif version == 2:
        visualize_grid_attention_v2(img_path=img_path,
                                   save_path=save_path,
                                   attention_mask=normed_attention_mask,
                                   save_image=True,
                                   save_original_image=True,
                                   quality=quality)

class RandomErasing(object):
    def __init__(self, EPSILON=0.5, sl=0.02, sh=0.33, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            print(img.size())
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[1]
                return img

        return img


class FARTrackDistill(BaseTracker):
    def __init__(self, params, dataset_name):
        super(FARTrackDistill, self).__init__(params)
        network = build_fartrack_distill(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        print("FARTrack model loaded from: ", self.params.checkpoint)
        self.cfg = params.cfg
        self.bins = self.cfg.MODEL.BINS
        self.network = network.cuda()
        self.network.eval()
        self.num_template = self.cfg.DATA.TEMPLATE.NUMBER
        self.preprocessor = Preprocessor()
        self.state = None
        self.update_ = False

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.use_visdom = params.debug
        self.frame_id = 0
        self.erase = RandomErasing()
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
        self.allnum = 0


    def template_update_sampling(self, new_z, sampling_method="linear"):
        # """
        # #   Update the template to support multiple sampling methods.
            
        # #   Args:
        # #     new_z: New template.
        # #     sampling_method: Sampling method, supporting the following options:
        # #       - "linear": Linear uniform sampling. # 95 762
        # #       - "exponential": Exponential decay sampling (denser for recent frames). 0.5 75.3 0.25 74.8 0.75 77 0.80 76.8 0.7 77.1
        # #       - "logarithmic": Logarithmic increasing sampling (denser for earlier frames). 76.1
        # #       - "random": Random sampling (the 0th frame and the latest frame are always retained).
        # #       - "fixed_weight": Fixed interval decreasing weight sampling.
        # """
        if not hasattr(self, 'stored_templates'):
            self.stored_templates = [] 
            self.stored_templates.append(self.z_dict1[0])  

        if self.frame_id >= 1:
            self.stored_templates.append(new_z)

        current_frame_count = self.frame_id + 1 
        num_templates = self.num_template  

        if current_frame_count < num_templates:
            for i in range(current_frame_count):
                self.z_dict1[i + 1] = self.stored_templates[i]
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

    def initialize(self, image, info: dict, name: str):
        # forward the template once
        current_file_path = os.path.abspath(__file__)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        
        self.root = os.path.join(project_root, "got10k_coor_mask")
        # self.root = f"/data5/FARTrack/main/fartrack_distill-base/got10k_coor_mask/"
        self.name = name
        
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

        self.state = info['init_bbox']
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
        #     os.mkdir(path+"/whole/")
        
        H, W, _ = image.shape
        self.frame_id += 1
        # if self.frame_id == 1:
        #     name = "template" + ".png"
        #     template = self.z_dict1[0].cpu().reshape(3, 112, 112).permute(1, 2, 0).numpy()
        #     template = np.clip(template*std+mean, 0, 1)*255
        #     im = Image.fromarray(np.uint8((template)))
        #     im.save(path + "/" + name)
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        # name = "search" + "_" + str(self.frame_id) + ".png"
        # searchx = search.tensors.cpu().reshape(3, 224, 224).permute(1, 2, 0).numpy()
        # searchx = np.clip(searchx*std+mean, 0, 1)*255
        # im = Image.fromarray(np.uint8((searchx)))
        # im.save(path + "/" + name)

        with torch.no_grad():
            x_dict = search

            out_dict = self.network.forward(
                template=self.z_dict1, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        pred_boxes = out_dict['seqs'][:, 0:4] / (self.bins - 1) - 0.5
        pred_boxes = pred_boxes.view(-1, 4).mean(dim=0)
        pred_new = pred_boxes

        pred_new[2] = pred_boxes[2] - pred_boxes[0]
        pred_new[3] = pred_boxes[3] - pred_boxes[1]
        pred_new[0] = pred_boxes[0] + pred_boxes[2] / 2
        pred_new[1] = pred_boxes[1] + pred_boxes[3] / 2

        pred_boxes = (pred_new * self.params.search_size / resize_factor).tolist()

        attention = out_dict['attention_list']
        layer_path = []
        rm_layer_path = []
        # for i in range(len(attention)):
        #     attn = attention[i][0].mean(dim=0)[-4:, 245:-4].mean(dim=0)
        #     search_path = path + "/" + name
        #     run_grid_attention_example(search_path, path + "/" + "search" + "_" + str(
        #         self.frame_id) + "_layer" + str(i), attn.cpu().numpy(), 2, 100)

        #     layer_path.append(path + "/" + "search" + "_" + str(self.frame_id) + "_layer" + str(i)+'/'+'search_'+str(self.frame_id)+"_with_attention.jpg")
        #     rm_layer_path.append(path + "/" + "search" + "_" + str(self.frame_id) + "_layer" + str(i)+'/')


        # original_image_path = search_path
        # template_path = path + "/template.png"
        # output_path = path +'/whole/' + "search" + "_" + str(self.frame_id) +'_whole.png'
        # concatenate_attention_maps(original_image_path, template_path, layer_path, output_path)
        # for i in rm_layer_path:
        #     shutil.rmtree(i)



        # get the final box result
        self.state = clip_box(self.map_box_back(pred_boxes, resize_factor), H, W, margin=10)
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, self.state, self.params.template_factor,
                                                                output_sz=self.params.template_size)  # (x1, y1, w, h)
        new_z = self.preprocessor.process(z_patch_arr, z_amask_arr).tensors   

        self.template_update_sampling(new_z, "exponential") # exponential

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
    return FARTrackDistill

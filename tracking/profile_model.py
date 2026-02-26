import argparse
import torch
import os
import sys
prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)
import _init_paths
from lib.utils.merge import merge_template_search
from thop import profile
from thop.utils import clever_format
import time
import importlib
from torch import nn



def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Parse args for training')
    # for train
    parser.add_argument('--script', type=str, default='fartrack_seq',
                        help='training script name')
    parser.add_argument('--config', type=str, default='fartrack_tiny_seq_224_full', help='yaml configure file name')
    args = parser.parse_args()

    return args

def evaluate(model, images_list, xz, run_box_head, run_cls_head, bs):
    """Compute FLOPs, Params, and Speed"""
    # # backbone
    macs1, params1 = profile(model, inputs=(images_list, None, "backbone", False, False), verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('backbone macs is ', macs)
    print('backbone params is ', params)
    # head
    macs2, params2 = profile(model, inputs=(None, xz, "head", True, True), verbose=False)
    macs, params = clever_format([macs2, params2], "%.3f")
    print('head macs is ', macs)
    print('head params is ', params)
    # the whole model
    macs, params = clever_format([macs1 + macs2, params1 + params2], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    '''Speed Test'''
    T_w = 100
    T_t = 1000
    print("testing speed ...")
    with torch.inference_mode():
        # overall
        for i in range(T_w):
            _ = model(images_list, None, "backbone", run_box_head, run_cls_head)
            _ = model(None, xz, "head", run_box_head, run_cls_head)
        # test
        min_latency = float('inf')
        for i in range(T_t):
            start = time.time()
            _ = model(images_list, None, "backbone", run_box_head, run_cls_head)
            _ = model(None, xz, "head", run_box_head, run_cls_head) 
            end = time.time()
            cur_latency = (end - start) / bs
            min_latency = min(min_latency, cur_latency)
            
        latency_ms = min_latency * 1000
        fps = 1000 / latency_ms
        print("The minimum overall latency is %.2f ms" % latency_ms)
        print("FPS: %.2f" % fps)


def evaluate_fartrack(model, template, dz_feat, search, seqs_input, bs=1):
    """Compute FLOPs, Params, and Speed"""
    # (template, search, _, _, _, seqs_input) 
    macs1, params1 = profile(model, inputs=(template, search, None, None, False, seqs_input),
                             custom_ops=None, verbose=False)
    macs, params = clever_format([macs1, params1], "%.3f")
    print('overall macs is ', macs)
    print('overall params is ', params)

    '''Speed Test'''
    T_w = 100
    T_t = 1000
    print("testing speed ...")
    with torch.no_grad():
        # overall
        for k in range(100):
            for i in range(T_w):
                _ = model(template, search, None, None, False, seq_input=seqs_input)
            start = time.time()
            for i in range(T_t):
                _ = model(template, search, None, None, False, seq_input=seqs_input)

            end = time.time()
            avg_lat = (end - start) / T_t
            print("The average overall latency is %.2f ms" % (avg_lat * 1000))
            print("FPS is %.2f fps" % (1. / avg_lat))



def get_data(bs, sz):
    img_patch = torch.randn(bs, 3, sz, sz)
    return img_patch

if __name__ == "__main__":
    # device = "cpu"
    
    device = "cuda:0"
    torch.cuda.set_device(device)
    # Compute the Flops and Params of our STARK-S model
    args = parse_args()
    '''update cfg'''
    yaml_fname = prj_path + '/experiments/%s/%s.yaml' % (args.script, args.config)
    config_module = importlib.import_module('lib.config.%s.config' % args.script)
    cfg = config_module.cfg
    config_module.update_config_from_file(yaml_fname)
    '''set some values'''
    bs = 1
    z_sz = cfg.TEST.TEMPLATE_SIZE
    x_sz = cfg.TEST.SEARCH_SIZE
    
    '''import vt network module'''
    model_module = importlib.import_module('lib.models.fartrack_seq')
    if args.script == "fartrack_seq":
        # torch.backends.cudnn.benchmark = False
        model_constructor = model_module.build_fartrack_seq
        model = model_constructor(cfg, training=False) # Train=False
        # get the template and search
        
        template = [torch.randn(bs, 3, z_sz, z_sz).to(device)]*1
        dz_feat = torch.randn(bs, 144, 192).to(device)
        search = get_data(bs, x_sz)
        # seqs_input = torch.randn(bs, 28) # 4*7 coordinates sequence
        seqs_input = torch.randint(200, 600, (bs, 4))
        # transfer to device
        model = model.to(device)
        dz_feat = dz_feat.to(device)
        search = search.to(device)
        seqs_input = seqs_input.to(device)
        model.eval()
        evaluate_fartrack(model, template, dz_feat, search, seqs_input, bs=bs)

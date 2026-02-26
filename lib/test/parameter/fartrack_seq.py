from lib.test.utils import TrackerParams
import os
from lib.test.evaluation.environment import env_settings
from lib.config.fartrack_seq.config import cfg, update_config_from_file


def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = env_settings().prj_dir
    save_dir = env_settings().save_dir
    # new add
    network_path = env_settings().network_path
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/fartrack_seq/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    # params.checkpoint = os.path.join(save_dir, "checkpoints/train/fartrack_seq/%s/FARTrackSeq_ep%04d.pth.tar" %
    #                                  (yaml_name, cfg.TEST.EPOCH))

    # tiny distill sparse checkpoints
    params.checkpoint = "/data5/FARTrack-re/main/fartrack_sparse/checkpoints/train/fartrack_sparse/fartrack_sparse_224_full/FARTrackSparse_ep0016.pth.tar" # AO:
    
    
    # whether to save boxes from all queries
    params.save_all_boxes = False

    return params 

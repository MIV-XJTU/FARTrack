import os
# loss function related
from lib.utils.box_ops import giou_loss
from torch.nn.functional import l1_loss
from torch.nn import BCEWithLogitsLoss
# train pipeline related
from lib.train.trainers import LTRTrainer, LTRSeqTrainer, LTRSeqTrainerV2, LTRSeqTrainerSparse
from lib.train.dataset import Lasot, Got10k, MSCOCOSeq, ImagenetVID, TrackingNet, SAV, VastTrack
from lib.train.dataset import Lasot_lmdb, Got10k_lmdb, MSCOCOSeq_lmdb, ImagenetVID_lmdb, TrackingNet_lmdb
from lib.train.data import sampler, opencv_loader, processing, LTRLoader, sequence_sampler, sequence_sampler_v2, sequence_sampler_v3
# distributed training related
from torch.nn.parallel import DistributedDataParallel as DDP
# some more advanced functions
from .base_functions import *
# network related
from lib.models.fartrack import build_fartrack
from lib.models.fartrack_seq import build_fartrack_seq
from lib.models.fartrack_distill import build_fartrack_distill
from lib.models.fartrack_sparse import build_fartrack_sparse
# forward propagation related
from lib.train.actors import FARTrackActor, FARTrackSeqActor, FARTrackDistillActor, FARTrackSparseActor
# for import modules
import importlib

from ..utils.focal_loss import FocalLoss

def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    #settings.use_lmdb = True
    for name in name_list:
        assert name in ["LASOT", "GOT10K_vottrain", "GOT10K_votval", "GOT10K_train_full", "GOT10K_official_val",
                        "COCO17", "VID", "TRACKINGNET", "SAV", "VastTrack"]
        if name == "LASOT":
            if settings.use_lmdb:
                print("Building lasot dataset from lmdb")
                datasets.append(Lasot_lmdb(settings.env.lasot_lmdb_dir, split='train', image_loader=image_loader))
            else:
                datasets.append(Lasot(settings.env.lasot_dir, split='train', image_loader=image_loader))
        if name == "GOT10K_vottrain":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='vottrain', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='vottrain', image_loader=image_loader))
        if name == "GOT10K_train_full":
            if settings.use_lmdb:
                print("Building got10k_train_full from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='train_full', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='train_full', image_loader=image_loader))
        if name == "GOT10K_votval":
            if settings.use_lmdb:
                print("Building got10k from lmdb")
                datasets.append(Got10k_lmdb(settings.env.got10k_lmdb_dir, split='votval', image_loader=image_loader))
            else:
                datasets.append(Got10k(settings.env.got10k_dir, split='votval', image_loader=image_loader))
        if name == "GOT10K_official_val":
            if settings.use_lmdb:
                raise ValueError("Not implement")
            else:
                datasets.append(Got10k(settings.env.got10k_val_dir, split=None, image_loader=image_loader))
        if name == "COCO17":
            if settings.use_lmdb:
                print("Building COCO2017 from lmdb")
                datasets.append(MSCOCOSeq_lmdb(settings.env.coco_lmdb_dir, version="2017", image_loader=image_loader))
            else:
                datasets.append(MSCOCOSeq(settings.env.coco_dir, version="2017", image_loader=image_loader))
        if name == "VID":
            if settings.use_lmdb:
                print("Building VID from lmdb")
                datasets.append(ImagenetVID_lmdb(settings.env.imagenet_lmdb_dir, image_loader=image_loader))
            else:
                datasets.append(ImagenetVID(settings.env.imagenet_dir, image_loader=image_loader))
        if name == "TRACKINGNET":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(TrackingNet(settings.env.trackingnet_dir, image_loader=image_loader))
        if name == "SAV":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(SAV(settings.env.sav_dir, image_loader=image_loader))
        if name == "VastTrack":
            if settings.use_lmdb:
                print("Building TrackingNet from lmdb")
                datasets.append(TrackingNet_lmdb(settings.env.trackingnet_lmdb_dir, image_loader=image_loader))
            else:
                # raise ValueError("NOW WE CAN ONLY USE TRACKINGNET FROM LMDB")
                datasets.append(VastTrack(settings.env.vasttrack_dir, split='train', image_loader=image_loader,
                                      multi_modal_vision=False,
                                      multi_modal_language=False))
    return datasets

def slt_collate(batch):
    ret = {}
    for k in batch[0].keys():
        here_list = []
        for ex in batch:
            here_list.append(ex[k])
        ret[k] = here_list
    return ret

class SLTLoader(torch.utils.data.dataloader.DataLoader):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    """

    __initialized = False

    def __init__(self, name, dataset, training=True, batch_size=1, shuffle=False, sampler=None, batch_sampler=None,
                 num_workers=0, epoch_interval=1, collate_fn=None, stack_dim=0, pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None):

        if collate_fn is None:
            collate_fn = slt_collate

        super(SLTLoader, self).__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                 num_workers, collate_fn, pin_memory, drop_last,
                 timeout, worker_init_fn)

        self.name = name
        self.training = training
        self.epoch_interval = epoch_interval
        self.stack_dim = stack_dim

def run(settings):
    settings.description = 'Training script for STARK-S, STARK-ST stage1, and STARK-ST stage2'

    # update the default configs with config file
    if not os.path.exists(settings.cfg_file):
        raise ValueError("%s doesn't exist." % settings.cfg_file)
    config_module = importlib.import_module("lib.config.%s.config" % settings.script_name)
    cfg = config_module.cfg
    config_module.update_config_from_file(settings.cfg_file)
    if settings.local_rank in [-1, 0]:
        print("New configuration is shown below.")
        for key in cfg.keys():
            print("%s configuration:" % key, cfg[key])
            print('\n')

    # update settings based on cfg
    update_settings(settings, cfg)

    # Record the training log
    log_dir = os.path.join(settings.save_dir, 'logs')
    if settings.local_rank in [-1, 0]:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    settings.log_file = os.path.join(log_dir, "%s-%s.log" % (settings.script_name, settings.config_name))

    # Build dataloaders

    if "RepVGG" in cfg.MODEL.BACKBONE.TYPE or "swin" in cfg.MODEL.BACKBONE.TYPE or "LightTrack" in cfg.MODEL.BACKBONE.TYPE:
        cfg.ckpt_dir = settings.save_dir
    bins = cfg.MODEL.BINS
    search_size = cfg.DATA.SEARCH.SIZE
    
    # Create network
    if settings.script_name == "fartrack":
        net = build_fartrack(cfg)
        loader_train, loader_val = build_dataloaders(cfg, settings)
    elif settings.script_name == "fartrack_seq":
        net = build_fartrack_seq(cfg)
        dataset_train = sequence_sampler_v2.SequenceSampler(
            datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
            max_gap=cfg.DATA.MAX_GAP, max_interval=cfg.DATA.MAX_INTERVAL,
            num_search_frames=cfg.DATA.SEARCH.NUMBER, num_template_frames=1,
            frame_sample_mode='random_interval',
            prob=cfg.DATA.INTERVAL_PROB)
        loader_train = SLTLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.TRAIN.NUM_WORKER,
                                 shuffle=False, drop_last=True)
    # fartrack_distill
    elif settings.script_name == "fartrack_distill": # Perform distillation based on FARTrack.
        net = build_fartrack_distill(cfg)
        loader_train, loader_val = build_dataloaders(cfg, settings)
    # fartrack_sparse
    elif settings.script_name == "fartrack_sparse":
        net = build_fartrack_sparse(cfg)
        dataset_train = sequence_sampler_v3.SequenceSampler(
            datasets=names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader),
            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
            samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
            max_gap=cfg.DATA.MAX_GAP, max_interval=cfg.DATA.MAX_INTERVAL,
            num_search_frames=cfg.DATA.SEARCH.NUMBER, num_template_frames=1,
            frame_sample_mode='random_interval',
            prob=cfg.DATA.INTERVAL_PROB)
        loader_train = SLTLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE,
                                 num_workers=cfg.TRAIN.NUM_WORKER,
                                 shuffle=False, drop_last=True)
    else:
        raise ValueError("illegal script name")

    # wrap networks to distributed one
    net.cuda()
    if settings.local_rank != -1:
        # net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)  # add syncBN converter
        net = DDP(net, device_ids=[settings.local_rank], find_unused_parameters=True)
        settings.device = torch.device("cuda:%d" % settings.local_rank)
    else:
        settings.device = torch.device("cuda:0")
    settings.deep_sup = getattr(cfg.TRAIN, "DEEP_SUPERVISION", False)
    settings.distill = getattr(cfg.TRAIN, "DISTILL", False)
    settings.distill_loss_type = getattr(cfg.TRAIN, "DISTILL_LOSS_TYPE", "KL")
    # Loss functions and Actors
    if settings.script_name == "fartrack":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 2., 'score': cfg.TRAIN.SCORE_WEIGHT}
        actor = FARTrackActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg, bins=bins, search_size=search_size)
    elif settings.script_name == "fartrack_seq":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 2., 'score_update': cfg.TRAIN.SCORE_WEIGHT}
        actor = FARTrackSeqActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg, bins=bins, search_size=search_size)
    elif settings.script_name == "fartrack_distill":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 2., 'score': cfg.TRAIN.SCORE_WEIGHT}
        actor = FARTrackDistillActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg, bins=bins, search_size=search_size)
    elif settings.script_name == "fartrack_sparse":
        focal_loss = FocalLoss()
        objective = {'giou': giou_loss, 'l1': l1_loss, 'focal': focal_loss}
        loss_weight = {'giou': cfg.TRAIN.GIOU_WEIGHT, 'l1': cfg.TRAIN.L1_WEIGHT, 'focal': 2., 'score_update': cfg.TRAIN.SCORE_WEIGHT}
        actor = FARTrackSparseActor(net=net, objective=objective, loss_weight=loss_weight, settings=settings, cfg=cfg, bins=bins, search_size=search_size)
    else:
        raise ValueError("illegal script name")

    # if cfg.TRAIN.DEEP_SUPERVISION:
    #     raise ValueError("Deep supervision is not supported now.")

    # Optimizer, parameters, and learning rates
    if settings.script_name == 'fartrack' or settings.script_name == 'fartrack_seq' or settings.script_name == 'fartrack_distill' or settings.script_name == 'fartrack_sparse':
        optimizer, lr_scheduler = get_optimizer_scheduler_fartrack(net, cfg)
    use_amp = getattr(cfg.TRAIN, "AMP", False)
    if settings.script_name == "fartrack":
        trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
    elif settings.script_name == "fartrack_seq":
        trainer = LTRSeqTrainerV2(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)
    elif settings.script_name == "fartrack_distill":
        trainer = LTRTrainer(actor, [loader_train, loader_val], optimizer, settings, lr_scheduler, use_amp=use_amp)
    elif settings.script_name == "fartrack_sparse":
        trainer = LTRSeqTrainerSparse(actor, [loader_train], optimizer, settings, lr_scheduler, use_amp=use_amp)

    # train process
    trainer.train(cfg.TRAIN.EPOCH, load_latest=True, fail_safe=True)

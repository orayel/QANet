from detectron2.config import CfgNode as CN


def add_qanet_config(cfg):

    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.MASK_ON = True
    cfg.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
    cfg.MODEL.PIXEL_STD = [58.395, 57.120, 57.375]

    # [QANET]
    cfg.MODEL.QANET = CN()

    # [Features Enhance Module]
    cfg.MODEL.FEATURES_ENHANCE = CN()
    cfg.MODEL.FEATURES_ENHANCE.RFENAME = "ASPP"
    cfg.MODEL.FEATURES_ENHANCE.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.FEATURES_ENHANCE.NUM_CHANNELS = 256

    # [Features Merging Module]
    cfg.MODEL.FEATURES_MERGING = CN()
    cfg.MODEL.FEATURES_MERGING.IS_USING_HAM = True

    # [Decoder]
    cfg.MODEL.DECODER = CN()
    cfg.MODEL.DECODER.HIDDEN_DIM = 128
    cfg.MODEL.DECODER.NUM_MASKS = 100
    cfg.MODEL.DECODER.GROUPS = 4
    cfg.MODEL.DECODER.SCALE_FACTOR = 2

    # [EprDetector]
    cfg.MODEL.EPRDETECTOR = CN()
    cfg.MODEL.EPRDETECTOR.DIM = 256
    cfg.MODEL.EPRDETECTOR.NUM_HEAD = 4
    cfg.MODEL.EPRDETECTOR.DIM_FORWARD = 1024
    cfg.MODEL.EPRDETECTOR.DROPOUT = 0.1
    cfg.MODEL.EPRDETECTOR.ACT = 'relu'
    cfg.MODEL.EPRDETECTOR.NUM_LAYER = 3

    # [Matcher]
    cfg.MODEL.MATCHER = CN()
    cfg.MODEL.MATCHER.ALPHA = 0.7
    cfg.MODEL.MATCHER.BETA = 0.2
    cfg.MODEL.MATCHER.GAMA = 0.1

    # [CRITERION]
    cfg.MODEL.CRITERION = CN()
    # cfg.MODEL.CRITERION.ITEMS = ("masks", "edges", "obj", "eprs")
    cfg.MODEL.CRITERION.ITEMS = ("masks", "edges", "obj")
    cfg.MODEL.CRITERION.LOSS_MASKS_DICE_WEIGHT = 1.0
    cfg.MODEL.CRITERION.LOSS_MASKS_BCE_WEIGHT = 1.0
    cfg.MODEL.CRITERION.LOSS_EDGES_DICE_WEIGHT = 1.0
    cfg.MODEL.CRITERION.LOSS_EDGES_BCE_WEIGHT = 1.0
    cfg.MODEL.CRITERION.LOSS_OBJ_WEIGHT = 1.0
    cfg.MODEL.CRITERION.LOSS_EPRS_WEIGHT = 1.0

    # [INFERENCE]
    cfg.MODEL.INFERENCE = CN()
    cfg.MODEL.INFERENCE.OBJ_THRESHOLD = 0.005
    cfg.MODEL.INFERENCE.MASK_THRESHOLD = 0.45
    cfg.MODEL.INFERENCE.MAX_DETECTIONS = 100

    # [Optimizer]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.AMSGRAD = False

    # [Dataset Mapper]
    cfg.MODEL.QANET.DATASET_MAPPER = "QANetInstDatasetMapper"

    # [INPUT]
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.RANDOM_FLIP = True
    cfg.INPUT.FORMAT = "RGB"
    cfg.INPUT.MASK_FORMAT = "bitmask"

    # [Pyramid Vision Transformer]
    cfg.MODEL.PVTV2 = CN()
    cfg.MODEL.PVTV2.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # [SWIN]
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]
    cfg.MODEL.SWIN.USE_CHECKPOINT = False

    # [OUTPUT]
    cfg.OUTPUT_DIR = 'output'
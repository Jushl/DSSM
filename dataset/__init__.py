from dataset.TDSDataset import TwoStageDecoupledStreamingDataset
from dataset.TDSTransform import TwoStageDecoupledStreamingTransforms
from dataset.TDSSampler import TwoStageDecoupledStreamingSampler
from dataset.TDSDataloader import TwoStageDecoupledStreamingDataloader


def build_TDSDataset(mode, args):
    transforms = TwoStageDecoupledStreamingTransforms(mode=mode, ttransforms=args.ttransforms, args=args)
    return TwoStageDecoupledStreamingDataset(mode, transforms, args)


EMRS_category2name = {
    0: 'car',
    1: 'two-wheel',
    2: 'pedestrian',
    3: 'bus',
    4: 'truck',
}

EMRS_name2category = {
    'car': 0,
    'two-wheel': 1,
    'pedestrian': 2,
    'bus': 3,
    'truck': 4,
}

VisDrone_category2name = {
    0: 'ignored',
    1: 'pedestrian',
    2: 'people',
    3: 'bicycle',
    4: 'car',
    5: 'van',
    6: 'truck',
    7: 'tricycle',
    8: 'awning-tricycle',
    9: 'bus',
    10: 'motor',
    11: 'others',
}

VisDrone_name2category = {
    'ignored': 0,
    'pedestrian': 1,
    'people': 2,
    'bicycle': 3,
    'car': 4,
    'van': 5,
    'truck': 6,
    'tricycle': 7,
    'awning-tricycle': 8,
    'bus': 9,
    'motor': 10,
    'others': 11,
}

EMRS_category2label = {k: i for i, k in enumerate(EMRS_category2name.keys())}
EMRS_label2category = {v: k for k, v in EMRS_category2label.items()}

VisDrone_category2label = {k: i for i, k in enumerate(VisDrone_category2name.keys())}
VisDrone_label2category = {v: k for k, v in VisDrone_category2label.items()}

from dataset.transforms import *


def TwoStageDecoupledStreamingTransforms(mode='train', ttransforms=False, args=None):
    if mode == 'train' and not ttransforms:
        return Compose(
            [
                Mosaic(imgsz=args.imgsz),
                RandomPhotometricDistort(p=0.5),
                RandomPerspective(translate=0.1, scale=0.5),
                RandomHorizontalFlip(p=0.5),
                RandomVerticalFlip(p=0.5),
                ConvertPILImage(dtype='float32', scale=True),
                ConvertBoxes(fmt='cxcywh', normalize=True)
            ]
        )
    if mode == 'train' and ttransforms:
        return Compose(
            [
                RandomPhotometricDistort(p=0.5),
                StreamingHorizontalFlip(),
                StreamingVerticalFlip(),
                ConvertPILImage(dtype='float32', scale=True),
                ConvertBoxes(fmt='cxcywh', normalize=True)
            ]
        )
    if mode == 'val' or mode == 'test':
        return Compose(
            [
                ConvertPILImage(dtype='float32', scale=True),
                ConvertBoxes(fmt='cxcywh', normalize=True)
             ]
        )
    raise ValueError(f'unknown {mode}')

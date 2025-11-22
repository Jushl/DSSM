import torch
import random
import math
import cv2
import torchvision
import numpy as np
from PIL import Image
from typing import Any, Dict
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from dataset.function import convert_to_tv_tensor


def box_clip(boxes, w, h):
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h)
    return boxes


def box_scale(boxes, scale_w, scale_h):
    scale = (scale_w, scale_h, scale_w, scale_h)
    boxes[:, 0] *= scale[0]
    boxes[:, 1] *= scale[1]
    boxes[:, 2] *= scale[2]
    boxes[:, 3] *= scale[3]
    return boxes


def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)


def remove_zero_area_boxes(boxes):
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    boxes_areas = widths * heights
    good = boxes_areas > 0
    if not all(good):
        boxes = boxes[good]
    return boxes, good


def horizontal_flip(image, target):
    image = F.horizontal_flip(image)
    if target['boxes'].any():
        w, h = image.size
        boxes = target["boxes"].copy()
        boxes = boxes[:, [2, 1, 0, 3]] * np.array([-1, 1, -1, 1]) + np.array([w, 0, w, 0])
        target["boxes"] = boxes
    return image, target


def vertical_flip(image, target):
    image = F.vertical_flip(image)
    if target['boxes'].any():
        w, h = image.size
        boxes = target["boxes"].copy()
        boxes = boxes[:, [0, 3, 2, 1]] * np.array([1, -1, 1, -1]) + np.array([0, h, 0, h])
        target["boxes"] = boxes
    return image, target


# 聚合transforms
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target, data):
        for t in self.transforms:
            image, target, data = t(image, target, data)
        data_idx = data['data_idx']
        return image, target, data_idx


class Mosaic:
    def __init__(self, imgsz=640, n=4):
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)
        self.n = n

    def __call__(self, image, target, data):
        data_structure = data.get("data_structure", None)
        mix = self.load_image(data_structure)
        image, target = self._mix_transform(image, target, mix)
        image = Image.fromarray(image)
        return image, target, data

    def load_image(self, data_structure):
        from dataset.TDSDataset import scale_transform
        from dataset.TDSDataset import TwoStageDecoupledStreamingDataset as TDSDataset

        _idxs = random.choices(range(len(data_structure)), k=self.n - 1)
        mix = []
        for _idx in _idxs:
            idx = random.randint(0, data_structure[_idx]['num'] - 1)
            image_path, anno_path = data_structure[_idx]['pairs'][idx]
            image = TDSDataset.load_image(image_path)
            target = TDSDataset.load_anno(anno_path, image, idx)
            image, target = scale_transform(image, target, self.imgsz)
            mix.append({
                'image': image,
                'target': target
            })
        return mix

    def _mix_transform(self, image, target, mix):
        return self._mosaic4(image, target, mix)

    def _mosaic4(self, image, target, mix):
        mosaic_targets = []
        s = self.imgsz
        yc, xc = (int(random.uniform(-x, 2 * s + x)) for x in self.border)
        for i in range(4):
            img = np.array(image) if i == 0 else np.array(mix[i - 1]['image'])
            tgt = target if i == 0 else mix[i - 1]['target']

            h, w = tgt.get('resized_shape', target['orig_shape'])
            if i == 0:  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b
            if tgt['boxes'].any():
                tgt = self._update_labels(tgt, padw, padh)
            mosaic_targets.append(tgt)
        final_target = self._cat_targets(mosaic_targets)
        return img4, final_target

    def _update_labels(self, tgt, padw, padh):
        offset = (padw, padh, padw, padh)
        tgt['boxes'][:, 0] += offset[0]
        tgt['boxes'][:, 1] += offset[1]
        tgt['boxes'][:, 2] += offset[2]
        tgt['boxes'][:, 3] += offset[3]
        return tgt

    def _cat_targets(self, mosaic_targets):
        cls = []
        instances = []
        imgsz = self.imgsz * 2  # mosaic imgsz
        for target in mosaic_targets:
            if target['boxes'].any():
                cls.append(target["labels"])
                instances.append(target["boxes"])

        final_targets = {
            "orig_shape": [mosaic_targets[i]["orig_shape"] for i in range(len(mosaic_targets))],
            "resized_shape": [imgsz, imgsz],
            "labels": np.concatenate(cls, 0) if len(cls) != 0 else np.empty((0, )),
            "boxes": np.concatenate(instances, 0) if len(instances) != 0 else np.empty((0, 4)),
            "mosaic_border": self.border,
        }

        final_targets['boxes'] = box_clip(final_targets['boxes'], imgsz, imgsz) \
            if final_targets['boxes'].any() else final_targets['boxes']
        final_targets['boxes'], good = remove_zero_area_boxes(final_targets["boxes"]) \
            if final_targets['boxes'].any() else (final_targets['boxes'], None)
        final_targets["labels"] = final_targets["labels"][good] \
            if good is not None else final_targets["labels"]

        final_targets['batch_idx'] = np.zeros(len(final_targets['labels']))
        return final_targets


class RandomPhotometricDistort:
    def __init__(self, brightness=(0.875, 1.125), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.05, 0.05), p=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.hue = hue
        self.saturation = saturation
        self.p = p

    def __call__(self, image, target, data):
        if torch.rand(1) >= self.p:
            return image, target, data
        image = self._transform(image, self._get_params(image))
        return image, target, data

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        if params["brightness_factor"] is not None:
            inpt = F.adjust_brightness(inpt, brightness_factor=params["brightness_factor"])
        if params["contrast_factor"] is not None and params["contrast_before"]:
            inpt = F.adjust_contrast(inpt, contrast_factor=params["contrast_factor"])
        if params["saturation_factor"] is not None:
            inpt = F.adjust_saturation(inpt, saturation_factor=params["saturation_factor"])
        if params["hue_factor"] is not None:
            inpt = F.adjust_hue(inpt, hue_factor=params["hue_factor"])
        if params["contrast_factor"] is not None and not params["contrast_before"]:
            inpt = F.adjust_contrast(inpt, contrast_factor=params["contrast_factor"])
        if params["channel_permutation"] is not None:
            inpt = F.permute_channels(inpt, permutation=params["channel_permutation"])
        return inpt

    def _generate_value(self, left: float, right: float) -> float:
        return torch.empty(1).uniform_(left, right).item()

    def _get_params(self, image) -> Dict[str, Any]:
        num_channels = 3 if image.mode == "RGB" else 1
        params: Dict[str, Any] = {
            key: self._generate_value(range[0], range[1]) if torch.rand(1) < self.p else None
            for key, range in [
                ("brightness_factor", self.brightness),
                ("contrast_factor", self.contrast),
                ("saturation_factor", self.saturation),
                ("hue_factor", self.hue),
            ]
        }
        params["contrast_before"] = bool(torch.rand(()) < 0.5)
        params["channel_permutation"] = torch.randperm(num_channels) if torch.rand(1) < self.p else None
        return params


class RandomPerspective:
    def __init__(self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border

    def __call__(self, image, target, data):
        cls = target['labels']
        border = target.pop("mosaic_border", self.border)
        image = np.array(image)
        self.size = image.shape[1] + border[1] * 2, image.shape[0] + border[0] * 2
        image, M, scale = self.affine_transform(image, border)
        target['resized_shape'] = image.shape[:2]
        image = Image.fromarray(image)

        if target['boxes'].any():
            boxes_orig = target.get("boxes").copy()
            boxes = self.apply_bboxes(boxes_orig, M)
            boxes_new = box_clip(boxes.copy(), *self.size)
            boxes_orig = box_scale(boxes_orig, scale_w=scale, scale_h=scale)
            i = box_candidates(box1=boxes_orig.T, box2=boxes_new.T)
            target['labels'] = cls[i]
            target['boxes'] = np.array(boxes_new[i])
            target['batch_idx'] = np.zeros(len(target['labels']))  # debug了 没问题

        return image, target, data

    def affine_transform(self, img, border):
        # Center
        C = np.eye(3, dtype=np.float32)
        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)
        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)
        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)
        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)
        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)
        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    def apply_bboxes(self, bboxes, M):
        n = len(bboxes)
        if n == 0:
            return bboxes
        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine
        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target, data):
        if torch.rand(1) >= self.p:
            return image, target, data

        image, target = horizontal_flip(image, target)
        return image, target, data


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target, data):
        if torch.rand(1) >= self.p:
            return image, target, data

        image, target = vertical_flip(image, target)
        return image, target, data


class ConvertPILImage:
    def __init__(self, dtype='float32', scale=True) -> None:
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image, target, data):
        image = F.pil_to_tensor(image)
        if self.dtype == 'float32':
            image = image.float()
        if self.scale:
            image = image / 255.

        image = tv_tensors.Image(image)
        return image, target, data


class ConvertBoxes:
    def __init__(self, fmt='cxcywh', normalize=True):
        self.fmt = fmt
        self.normalize = normalize

    def __call__(self, image, target, data):
        boxes = torch.Tensor(target.get("boxes").copy())
        spatial_size = target.get("resized_shape")

        if not target['boxes'].any():
            boxes = convert_to_tv_tensor(boxes, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)
            target["boxes"] = boxes
            return image, target, data

        if self.fmt:
            boxes = torchvision.ops.box_convert(boxes, in_fmt="xyxy", out_fmt=self.fmt.lower())
            boxes = convert_to_tv_tensor(boxes, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)
        if self.normalize:
            boxes = boxes / torch.tensor(spatial_size[::-1]).tile(2)[None]

        target["boxes"] = boxes
        return image, target, data


class StreamingHorizontalFlip:
    def __call__(self, image, target, data):
        _decision = data.get("decision")["horizontal_flip"]
        should_flip = _decision[data.get("data_idx")[0]]  # 获取翻转决策
        if not should_flip:
            return image, target, data

        image, target = horizontal_flip(image, target)
        return image, target, data


class StreamingVerticalFlip:
    def __call__(self, image, target, data):
        _decision = data.get("decision")["vertical_flip"]
        should_flip = _decision[data.get("data_idx")[0]]  # 获取翻转决策
        if not should_flip:
            return image, target, data

        image, target = vertical_flip(image, target)
        return image, target, data




import random
import os
import json
import math
import numpy as np
import cv2
import dataset
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
from dataset.TDSTransform import TwoStageDecoupledStreamingTransforms


class TwoStageDecoupledStreamingDataset(Dataset):
    def __init__(self, mode, transforms, args):
        self.dataset_path = args.dataset_path
        self.mode = mode
        self.args = args
        if mode == "train":
            self.base_path = os.path.join(args.dataset_path, "train")
            self.batch_size = args.batch_size_train
        elif mode == "val":
            self.base_path = os.path.join(args.dataset_path, "val")
            self.batch_size = args.batch_size_val
        else:
            assert mode in ["train", "val"]

        self.ttransforms = args.ttransforms
        self.imgsz = args.imgsz
        self._transforms = transforms
        self.data_structure = []
        self._build_data_structure()
        self._build_flip_decision()

    def __len__(self):
        return sum(data['num'] for data in self.data_structure)

    def __getitem__(self, idx):
        _idx = 0
        while idx >= self.data_structure[_idx]['num']:
            idx -= self.data_structure[_idx]['num']
            _idx += 1
        data_idx = [_idx, idx]
        image_path, anno_path = self.data_structure[_idx]['pairs'][idx]
        data = {
            'data_idx': data_idx,
            'data_structure': self.data_structure,
            'decision': self._decision
        }

        image = self.load_image(image_path)
        target = self.load_anno(anno_path, image, idx)

        image, target = scale_transform(image, target, self.imgsz)
        image, target, data_idx = self._transforms(image, target, data)

        return image, target, data_idx

    def _build_data_structure(self):
        images_base_path = os.path.join(self.base_path, "images")
        for scene in sorted(os.listdir(images_base_path)):
            scene_path = os.path.join(images_base_path, scene)
            if os.path.isdir(scene_path):
                all_files = glob(os.path.join(scene_path, '*'))
                images_files = sorted([f for f in all_files
                                       if os.path.splitext(f)[1][1:].lower() in ['png', 'jpg', 'jpeg']])
                data_structure = []
                for image_path in images_files:
                    base_name = os.path.splitext(os.path.basename(image_path))[0]

                    label_path = os.path.join(scene_path.replace('images', 'labels'), f"{base_name}.json")
                    data_structure.append((image_path, label_path))
                if data_structure:
                    self.data_structure.append({
                        'pairs': data_structure,
                        'num': len(data_structure)
                    })

        random.shuffle(self.data_structure)

    def _build_flip_decision(self):
        num_decision = len(self.data_structure)
        if self.ttransforms:
            self._decision = {
                "vertical_flip": {i: random.random() < 0.5 for i in range(num_decision)},
                "horizontal_flip": {i: random.random() < 0.5 for i in range(num_decision)},
            }
        else:
            self._decision = {}

    def start_temporal(self):
        self.ttransforms = True
        if self.mode == "train":
            self._transforms = TwoStageDecoupledStreamingTransforms(ttransforms=self.ttransforms, args=self.args)
        print(f'{self.mode} start temporal')

    @staticmethod
    def load_image(image_path):
        image = Image.open(image_path).convert("RGB")
        return image

    @staticmethod
    def load_anno(anno_path, image, idx):
        W, H = image.size
        if os.path.exists(anno_path):
            anno = get_json_boxes(anno_path)
            target = {'image_id': idx, 'boxes': anno['boxes'], 'labels': anno['labels']}
            target = prepare(image, target)
        else:
            target = {
                'orig_shape': [H, W],
                'boxes': np.empty((0, 4)),
                'labels': np.empty((0, )),
                'image_id': idx,
                'batch_idx': np.empty((0, )),
            }
        return target


def get_json_boxes(label_filename):
    with open(label_filename, 'r') as json_file:
        data = json.load(json_file)
        objects = data['shapes']
        class_indexes = []
        bounding_boxes = []
        for i in range(len(objects)):
            bounding_boxes_points = objects[i]['points']
            if 'label' in objects[i]:
                bounding_boxes_class = objects[i]['label']
            else:
                bounding_boxes_class = objects[i]['label']

            folders = os.path.normpath(label_filename).split(os.sep)
            if 'VisDrone-VID' in folders:
                class_index = int(dataset.VisDrone_name2category[bounding_boxes_class])
            else:
                class_index = int(dataset.EMRS_name2category[bounding_boxes_class])

            bounding_box = [int(bounding_boxes_points[0][0]), int(bounding_boxes_points[0][1]),
                            int(bounding_boxes_points[2][0]), int(bounding_boxes_points[2][1])]

            class_indexes.append(class_index)
            bounding_boxes.append(bounding_box)

    return {'labels': class_indexes, 'boxes': bounding_boxes}


def prepare(image, target):
    w, h = image.size
    gt = {}
    gt["orig_shape"] = [int(h), int(w)]
    image_id = target["image_id"]
    boxes = target['boxes']
    classes = target['labels']
    boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
    boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, w)
    boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
    classes = np.array(classes, dtype=np.int64)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    classes = classes[keep]

    gt["boxes"] = boxes
    gt["labels"] = classes
    gt["image_id"] = image_id
    gt["batch_idx"] = np.zeros(len(classes))
    return gt


def scale_transform(image, target, imgsz, rect_mode=True):
    h0, w0 = target['orig_shape']
    image = np.array(image)
    if rect_mode:
        r = imgsz / max(h0, w0)
        if r != 1:
            w, h = (min(math.ceil(w0 * r), imgsz), min(math.ceil(h0 * r), imgsz))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    elif not (h0 == w0 == imgsz):
        image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)

    resized_h, resized_w = image.shape[:2]
    target['resized_shape'] = (resized_h, resized_w)
    target["ratio_pad"] = (
        resized_h / target["orig_shape"][0],
        resized_w / target["orig_shape"][1]
    )
    if target['boxes'].any():
        target['boxes'] = target['boxes'] / np.array([w0, h0, w0, h0]) \
                          * np.array([resized_w, resized_h, resized_w, resized_h])

    image = Image.fromarray(image)
    return image, target

from torch.utils.data import DataLoader
import torch
from dataset.collate_fn import BatchImageCollateFuncion
from dataset import TwoStageDecoupledStreamingSampler as TDSSampler
import random


class TwoStageDecoupledStreamingDataloader:
    def __init__(self, mode, dataset, args):
        self.dataset = dataset
        self.mode = mode
        self.epoch = args.epoch
        self.epochs = args.epochs
        self.tepochs = args.tepochs
        self.args = args
        self.init_dataloader()

    def init_dataloader(self, trainer=True):
        if self.check_temporal_epoch():
            self.dataloader = self.build_temporal_dataloader(trainer)
        else:
            if not hasattr(self, 'dataloader'):
                self.dataloader = self.build_spatial_dataloader()

    def check_temporal_epoch(self):
        return self.epoch == (self.epochs - self.tepochs)

    def build_spatial_dataloader(self):
        return DataLoader(
            dataset=self.dataset,
            sampler=torch.utils.data.RandomSampler(self.dataset),
            collate_fn=BatchImageCollateFuncion(),
            batch_size=self.args.batch_size_train if self.mode == "train" else self.args.batch_size_val,
            num_workers=self.args.num_workers,
            drop_last=True if self.mode == "train" else False,
            pin_memory=True if self.mode == "train" else False
        )

    def build_temporal_dataloader(self, trainer):
        if trainer:
            if self.args.batch_size_train_temporal != 1:
                for _ in range(self.args.batch_size_train_temporal - len(
                        self.dataset.data_structure) % self.args.batch_size_train_temporal):
                    self.dataset.data_structure.append(random.choice(self.dataset.data_structure))
        return DataLoader(
            dataset=self.dataset,
            batch_sampler=TDSSampler(self.dataset, self.args.batch_size_train_temporal if self.mode == "train" else self.args.batch_size_val),
            collate_fn=BatchImageCollateFuncion(),
            num_workers=self.args.num_workers,
            pin_memory=True if self.mode == "train" else False
        )

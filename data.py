# -*- coding: utf-8 -*-
import os
import sys
import copy
import random
import json
from loguru import logger
import numpy as np
from PIL import Image
from collections import Counter
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
import utils

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)


class DatasetWithIndicesWrapper(torch.utils.data.Dataset):
    def __init__(
        self,
        name,
        data,
        targets,
        cname2ix,
        extra=None,
        base_transforms=None,
        all_transforms=None,
    ):
        self.name = name
        self.data = data
        self.targets = targets
        self.cname2ix = cname2ix
        self.extra = extra
        self.base_transforms = base_transforms
        self.all_transforms = all_transforms        

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        data, target = self.data[index], self.targets[index]
        data = utils.default_loader(self.data[index])

        if self.base_transforms:
            data_og = self.base_transforms(data)
            data_aug1 = self.all_transforms(data)
            data_aug2 = self.all_transforms(data)
        if self.extra:
            extra = list(self.extra[index])
            extra = tuple(extra)

        return ((data_og, data_aug1, data_aug2), int(target), int(index), extra)


class UDADataset:
    """
    Dataset Class
    """

    def __init__(
        self,
        benchmark,
        name,
        is_target=False,
        dist=False,
    ):
        self.benchmark = benchmark
        self.name = name
        self.is_target = is_target
        self.num_classes = None
        self.cname2ix = {}
        self.dist = dist

    def get_num_classes(self):
        return self.num_classes

    def get_dataset(self):
        """Generates and returns dataset"""
        json_file = self.name
        if self.benchmark == "dollarstreet":
            json_path = os.path.join(
                "data", self.benchmark, "{}.json".format(json_file)
            )
        elif self.benchmark == "geoyfcc":
            json_path = os.path.join("data", "{}.json".format(json_file))
            id2name = os.path.join("data", "geoyfccid2name.json")
            with open(id2name, "r") as f:
                id2name = json.load(f)
        else:
            raise NotImplementedError

        logger.info("Loading {}...".format(json_path))
        with open(json_path, "r") as f:
            info_json = json.load(f)
            data, targets, countries, img_ids = [], [], [], []
            for category, v in info_json.items():
                for entry in v:
                    if self.benchmark == "dollarstreet":
                        fpath = os.path.join(
                            "data",
                            self.benchmark,
                            "images",
                            category,
                            "{}.jpg".format(entry["img_id"]),
                        )

                    elif self.benchmark == "geoyfcc":
                        fpath = os.path.join(
                            "data",
                            "{}".format(self.benchmark),
                            id2name[category],
                            "{}.jpg".format("".join(entry["filekey"].split("_"))),
                        )
                    else:
                        raise NotImplementedError

                    data.append(fpath)
                    targets.append(category)
                    countries.append(entry["country"])

                    if self.benchmark == "dollarstreet":
                        img_ids.append(entry["img_id"])
                    else:
                        img_ids.append(entry["filekey"])

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        train_transforms = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
                transforms.RandomApply(
                    # not strengthened
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([utils.GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        test_transforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        cnames = np.unique(targets)
        cnames.sort()
        self.num_classes = len(cnames)
        self.cname2ix = {cname: ix for ix, cname in enumerate(cnames)}
        targets = [self.cname2ix[t] for t in targets]

        self.dataset = DatasetWithIndicesWrapper(
            self.name,
            data,
            targets,
            self.cname2ix,
            tuple(zip(countries, img_ids)),
            test_transforms,
            train_transforms,
        )

        return self.dataset

    def get_loaders(
        self, test_ratio=0.1, class_balance_train=False, batch_size=64, subsample_size=0
    ):
        """Constructs and returns dataloaders
        Args:
                test_ratio (float): Proportion of data to hold out for testing
                class_balance_train (bool, optional): Whether to class-balance train data loader. Defaults to False.
        Returns:
                Train, val, test dataloaders, as well as selected indices used for training
        """
        if not self.dataset:
            self.get_dataset()
        indices = list(range(len(self.dataset)))
        test_idx = None
        if test_ratio > 0.0:
            split = int(np.floor(test_ratio * len(indices)))
            np.random.seed(1234)  # Deterministic splits
            np.random.shuffle(indices)
            train_idx, test_idx = indices[split:], indices[:split]
            test_dataset = copy.deepcopy(self.dataset)
            test_sampler = SubsetRandomSampler(test_idx)
        else:
            train_idx = indices

        if subsample_size > 0:
            assert test_ratio > 0, "Subsample supported only when test_ratio>0"
            train_idx = np.random.choice(train_idx, subsample_size, replace=False)

        if class_balance_train:
            self.dataset.data = [self.dataset.data[idx] for idx in train_idx]
            self.dataset.targets = np.array(self.dataset.targets)[train_idx]
            self.dataset.extra = [self.dataset.extra[idx] for idx in train_idx]

            if hasattr(self.dataset, "actual_targets"):
                self.dataset.actual_targets = np.array(self.dataset.actual_targets)[
                    train_idx
                ]

            targets = copy.deepcopy(self.dataset.targets)
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
            count_dict = Counter(targets)

            count_dict_full = {
                self.cname2ix[lbl]: 0 for lbl in list(self.cname2ix.keys())
            }
            for k, v in count_dict.items():
                count_dict_full[k] = v

            count_dict_sorted = {
                k: v
                for k, v in sorted(count_dict_full.items(), key=lambda item: item[0])
            }
            class_sample_count = np.array(list(count_dict_sorted.values()))
            class_sample_count = class_sample_count / class_sample_count.max()
            class_sample_count += 1e-8

            weights = 1 / torch.Tensor(class_sample_count)
            # print(weights)
            sample_weights = [weights[l] for l in targets]
            sample_weights = torch.DoubleTensor(np.array(sample_weights))
            train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        else:
            train_sampler = SubsetRandomSampler(train_idx)

        if self.dist:
            train_sampler = utils.DistributedSamplerWrapper(train_sampler)

        train_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=train_sampler,
            batch_size=batch_size,
            num_workers=6,
            shuffle=(train_sampler is None),
        )

        test_loader = torch.utils.data.DataLoader(
            self.dataset,
            sampler=SubsetRandomSampler(train_idx),
            batch_size=batch_size,
            pin_memory=True,
            num_workers=6,
            shuffle=None,
        )
        if test_ratio > 0.0:
            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                sampler=test_sampler,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=6,
                shuffle=(test_sampler is None),
            )

        return train_loader, test_loader, train_idx, test_idx

import os
import copy
import contextlib
import random
import json
import time
from collections import defaultdict, Counter, OrderedDict
import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger
from typing import Iterator, Optional
from operator import itemgetter
from model import ResNet50

import matplotlib
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image, ImageFilter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import init
import torch.nn.functional as F
from torch.autograd import Function

# from sentence_transformers import SentenceTransformer
import distributed as dist

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

######################################################################
# Data augmentation utilities
######################################################################


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


######################################################################
# Data loading utilities
######################################################################


def default_loader(path, rgb=True):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        with Image.open(f) as img:
            if rgb:
                return img.convert("RGB")
            else:
                return Image.fromarray(np.array(img))


def log(target_accs, fname, args):
    """
    Logs results JSON in a dedicated directory in logs/<source>2<target>/<exp_name>.json
    """
    benchmark = "{}2{}".format(args.source, args.target)
    os.makedirs(os.path.join("logs", benchmark), exist_ok=True)
    with open(os.path.join("logs", benchmark, "perf_{}.json".format(fname)), "w") as f:
        json.dump(target_accs, f, indent=4)


class DatasetFromSampler(torch.utils.data.Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:
                                    sampler: PyTorch sampler
    """

    def __init__(self, sampler: torch.utils.data.Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args:
                                        index: index of the element in the dataset
        Returns:
                                        Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """
        Returns:
                                        int: length of the dataset
        """
        return len(self.sampler)


class DistributedSamplerWrapper(torch.utils.data.DistributedSampler):
    """
    Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
                                    Sampler is assumed to be of constant size.
    """

    def __init__(
        self,
        sampler,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
    ):
        """
        Args:
                                        sampler: Sampler used for subsampling
                                        num_replicas (int, optional): Number of processes participating in
                                          distributed training
                                        rank (int, optional): Rank of the current process
                                          within ``num_replicas``
                                        shuffle (bool, optional): If true (default),
                                          sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(
            DatasetFromSampler(sampler),
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
        )
        self.sampler = sampler

    def __iter__(self) -> Iterator[int]:
        """Iterate over sampler.
        Returns:
                                        python iterator
        """
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

######################################################################
# Custom layers and losses
######################################################################

class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model

    def _encode_one_hot(self, ids, use_logits=True):
        if use_logits:
            one_hot = torch.zeros_like(self.logits).to(self.device)
        else:
            one_hot = torch.zeros_like(self.feats).to(self.device)
        one_hot[:, ids] = 1
        return one_hot

    def forward(self, image, get_feats, get_proj):
        retval = self.model(image, get_feats=get_feats, get_proj=get_proj)
        self.logits = retval[0] if isinstance(retval, tuple) else retval
        return retval

    def backward(self, ids):
        raise NotImplementedError

    def generate(self):
        raise NotImplementedError

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg()
        return output, None


def get_discriminator(num_classes):
    return nn.Sequential(
        nn.Linear(num_classes, 500),
        nn.ReLU(),
        nn.Linear(500, 500),
        nn.ReLU(),
        nn.Linear(500, 2),
    )


######################################################################
# Optimization utilities
######################################################################
def generate_optimizer(model, args, mode="da", params_list=None):
    lr = args.lr if mode == "source" else args.da_lr
    wd = args.wd if mode == "source" else args.da_wd

    if params_list is None:
        params_list = model.parameters()
    if args.optimizer == "Adam":
        optimizer = optim.Adam(
            params_list,
            lr=lr,
            weight_decay=wd,
            betas=(0.9, 0.999),
            eps=1e-08,
        )
    elif args.optimizer == "SGD":
        optimizer = optim.SGD(
            params_list, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True
        )
    else:
        raise NotImplementedError

    return optimizer


def inv_lr_scheduler(
    param_lr, optimizer, iter_num, gamma=0.0001, power=0.75, init_lr=0.001
):
    """
    Decay learning rate
    """
    for i, param_group in enumerate(optimizer.param_groups):
        lr = param_lr[i] * (1 + gamma * iter_num) ** (-power)
        param_group["lr"] = lr
    return optimizer


######################################################################
# Training utilities
######################################################################


def train(
    model,
    device,
    train_loader,
    optim,
    epoch,
    args,
):
    model.train()
    total_loss, avg_loss = 0.0, 0.0
    correct, total_el = 0.0, 0.0
    lambda_ft = 1.0


    batch_time_meter = AverageMeter("Time", ":6.3f")
    data_time_meter = AverageMeter("Data", ":6.3f")
    train_losses = AverageMeter("train loss", ":.2f")
    train_accs = AverageMeter("train acc.", ":2.2f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time_meter, data_time_meter, train_losses, train_accs],
        prefix=f"Epoch: [{epoch}]",
    )

    start_time = time.perf_counter()

    for batch_idx, (
        (data_og, data, data2),
        target,
        _,
        (_, img_ids),
    ) in enumerate(train_loader):
        data_time = time.perf_counter() - start_time

        data, target = data.to(device, non_blocking=True), target.to(
            device, non_blocking=True
        )
        optim.zero_grad()
        output = model(data)
        ce_loss = lambda_ft * nn.CrossEntropyLoss()(output, target)
        loss = ce_loss
        pred = output.argmax(dim=1, keepdim=True)
        correct_batch = pred.eq(target.view_as(pred)).sum().item()
        correct += correct_batch
        out_str = "Batch {:d}: CE loss: {:.3f}".format(batch_idx, ce_loss.item())

        total_loss += loss.item()
        loss.backward()
        optim.step()
        total_el += data.shape[0]

        # measure elapsed time
        batch_time = time.perf_counter() - start_time

        # update all progress meters
        data_time_meter.update(data_time)
        batch_time_meter.update(batch_time)
        train_losses.update(loss.item(), data.shape[0])
        train_accs.update(100.0 * correct_batch / data.shape[0], data.shape[0])

        if batch_idx % 10 == 0:
            logger.info(out_str)

        start_time = time.perf_counter()

    progress.display(batch_idx)
    train_acc = 100.0 * correct / total_el
    train_loss = total_loss / batch_idx
    return train_acc, train_loss


def test(model, device, test_loader, split="test", num_classes=10):
    model.eval()
    correct, total_el = 0, 0
    confusion_matrix = torch.zeros(num_classes, num_classes).long()

    embedding = torch.zeros([len(test_loader.dataset), num_classes])
    labels = torch.zeros(len(test_loader.dataset)).long()

    with torch.no_grad():
        for batch_idx, ((data_og, _, _), target, indices, _) in enumerate(test_loader):
            data_og, target = data_og.to(device, non_blocking=True), target.to(
                device, non_blocking=True
            )
            output = model(data_og)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total_el += data_og.shape[0]

            embedding[indices, :] = output.cpu()
            labels[indices] = target.cpu()

            for t, p in zip(target.view(-1), pred.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1

        nz_idxs = torch.arange(len(test_loader.dataset))[embedding.sum(dim=1) != 0]
        embedding = embedding[nz_idxs]
        labels = labels[nz_idxs]

    test_acc = 100.0 * correct / total_el
    brec_dict = get_balanced_recall_dict(embedding, labels, K=5)
    return test_acc, confusion_matrix, brec_dict


def train_source_model(
    source_model,
    src_train_loader,
    src_test_loader,
    num_classes,
    args,
    device,
):
    source_model.to(device)

    source_optimizer = generate_optimizer(source_model, args, mode="source")


    best_pca = 0.0

    for epoch in range(args.num_epochs):
        if dist.get_world_size() > 1:
            src_train_loader.sampler.set_epoch(epoch)
        train_acc, train_loss = train(
            source_model,
            device,
            src_train_loader,
            source_optimizer,
            epoch,
            args,
        )

        if dist.is_master_process():
            eval_model = (
                source_model.module if dist.get_world_size() > 1 else source_model
            )
            test_acc, cm, _ = test(
                eval_model,
                device,
                src_test_loader,
                split="test",
                num_classes=num_classes,
            )
            test_pca = (
                cm.diagonal().numpy() / (cm.sum(axis=1).numpy() + 1e-8)
            ).mean() * 100
            out_str = "Epoch {}:  Train Loss={:.3f}\tTrain Acc={:.2f}\tTest Acc={:.2f}\tTest pca={:.2f}".format(
                epoch,
                train_loss,
                train_acc,
                test_acc,
                test_pca,
            )
            logger.info(out_str)

            if test_pca > best_pca:
                best_pca = test_pca
                source_file_full = "{}_{}_source_{}_lm{}.pth".format(
                    args.source, args.cnn, args.source_id, args.use_lm
                )
                save_model = (
                    source_model.module if dist.get_world_size() > 1 else source_model
                )
                torch.save(
                    save_model.state_dict(),
                    os.path.join("checkpoints", "source", source_file_full),
                )


def get_embedding(model, loader, device, num_classes):
    model.eval()

    embedding = torch.zeros([len(loader.dataset), 2048])
    labels = torch.zeros(len(loader.dataset)).long()
    preds = torch.zeros(len(loader.dataset)).long()
    image_ids = np.array(["" for _ in range(len(loader.dataset))], dtype=object)

    with torch.no_grad():
        for batch_idx, ((data, _, _), target, indices, (_, img_ids, _)) in enumerate(
            tqdm(loader)
        ):
            data = data.to(device)
            scores, emb = model(data, get_feats=True)
            embedding[indices, :] = emb.cpu()
            labels[indices] = target
            preds[indices] = scores.argmax(dim=1, keepdim=True).squeeze().cpu()
            image_ids[indices] = img_ids

    nz_idxs = torch.arange(len(loader.dataset))[embedding.sum(dim=1) != 0]
    embedding = embedding[nz_idxs]
    labels = labels[nz_idxs]
    preds = preds[nz_idxs]
    image_ids = image_ids[nz_idxs]

    return embedding, labels, preds, image_ids


def get_balanced_recall_dict(outputs, targets, K=5):
    brec_dict = defaultdict(int)
    test_counter = Counter(targets.cpu().numpy())
    for ix in range(len(targets)):
        _, topk_pred = outputs[ix].topk(K, 0, True, True)

        if targets[ix] in topk_pred:
            brec_dict[targets[ix].item()] += 1

    for k, v in test_counter.items():
        if v == 0:
            brec_dict[k] = 0
            continue
        brec_dict[k] = float(brec_dict[k]) / test_counter[k]

    return brec_dict


######################################################################
# Plotting and logging utilities
######################################################################


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """
    Computes the recall@k for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def plot_cm(
    ax,
    conf_matrix,
    args,
    label="Confusion matrix",
    num_classes=10,
    class_dict=None,
    normalize=False,
):
    import seaborn as sns

    if normalize:
        conf_matrix = conf_matrix.float()
        conf_matrix /= conf_matrix.sum(dim=1)

    sns.set(font_scale=0.2)
    if class_dict is not None:
        inv_class_dict = {v: k for k, v in class_dict.items()}
        df_cm = pd.DataFrame(
            conf_matrix.cpu().numpy(),
            index=[inv_class_dict[i] for i in range(num_classes)],
            columns=[inv_class_dict[i] for i in range(num_classes)],
        )
    else:
        df_cm = pd.DataFrame(
            conf_matrix.cpu().numpy(),
            index=[i for i in range(num_classes)],
            columns=[i for i in range(num_classes)],
        )
    if num_classes > 10:
        hm = sns.heatmap(
            df_cm,
            annot=False,
            fmt="d",
            ax=ax,
            cbar=False,
            xticklabels=True,
            yticklabels=True,
        )
    else:
        hm = sns.heatmap(
            df_cm,
            annot=True,
            fmt="d",
            ax=ax,
            cbar=False,
            xticklabels=True,
            yticklabels=True,
        )
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=10)
    hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=10)
    ax.set_title(label, fontsize=18)
    ax.set_xlabel("Prediction", fontsize=16)
    ax.set_ylabel("Ground Truth", fontsize=16)


def plot_per_class_diff(
    ax,
    per_class_after,
    per_class_before,
    num_classes,
    train_loader,
    title,
    class_dict=None,
):
    tgts = train_loader.dataset.targets
    if isinstance(tgts, torch.Tensor):
        tgts = tgts.numpy()
    counts = Counter(tgts)

    sorted_counts = {
        ix: (k, v)
        for ix, (k, v) in enumerate(
            sorted(counts.items(), key=lambda item: item[1], reverse=True)
        )
    }
    order = np.array(list(sorted_counts.keys()))
    X = np.array([ix for ix in range(num_classes)])

    ax.bar(
        X,
        per_class_after[order] - np.array(per_class_before)[order],
        color="#4389E5",
        alpha=0.8,
    )
    inv_class_dict = {v: k for k, v in class_dict.items()}
    ax.set_xticks(X)
    ax.set_xticklabels([inv_class_dict[ix] for ix in order], fontsize=10, rotation=90)
    ax.set_ylim(-1.0, 1.0)
    ax.set_yticklabels(
        [np.round(ytick * 100, 2) for ytick in ax.get_yticks()], fontsize=18
    )
    ax.set_xlabel(r"Ground truth label (size decreases $\rightarrow$)", fontsize=22)
    ax.set_ylabel("Accuracy change (%)", fontsize=22)
    ax.set_xlim(-0.5, num_classes - 0.5)
    ax.set_title(title, fontsize=22)


def plot_accuracy_statistics(
    cm_before, cm_after, num_classes, args, target_train_loader, class_dict, exp_name
):
    pca_before = cm_before.diagonal().numpy() / cm_before.sum(axis=1).numpy()
    pca_after = cm_after.diagonal().numpy() / cm_after.sum(axis=1).numpy()

    matplotlib.rcParams.update({"font.size": 16})
    fig, axs = plt.subplots(1, 3, figsize=(22, 7))

    plot_cm(
        axs[0],
        cm_before,
        args,
        label="Before {} (Avg. acc: {:.2f})".format(
            args.da_strat, pca_before.mean() * 100
        ),
        num_classes=num_classes,
        class_dict=class_dict,
        normalize=True,
    )
    plot_cm(
        axs[1],
        cm_after,
        args,
        label="After {} (Avg. acc: {:.2f})".format(
            args.da_strat, pca_after.mean() * 100
        ),
        num_classes=num_classes,
        class_dict=class_dict,
        normalize=True,
    )
    plot_per_class_diff(
        axs[2],
        pca_after,
        pca_before,
        num_classes,
        target_train_loader,
        title="Per-class accuracy change",
        class_dict=class_dict,
    )

    fig.suptitle(
        r"{}$\rightarrow${}: {}".format(args.source, args.target, args.da_strat),
        fontsize=28,
        y=1.05,
    )
    # plt.tight_layout()
    plt.savefig(
        "results/{}_acc_analysis.png".format(exp_name),
        bbox_inches="tight",
        dpi=400,
    )

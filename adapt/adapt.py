# -*- coding: utf-8 -*-
import os
import sys
import random
import pickle
import copy
import json
from loguru import logger
import wandb
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import distributed as dist

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
from RandAugment import RandAugment

# from sentence_transformers import SentenceTransformer

from .solver import register_solver
import utils

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)


class BaseSolver:
    def __init__(self, net, src_loader, tgt_loader, tgt_opt, device, num_classes, args):
        self.net = net
        self.src_loader = src_loader
        self.tgt_loader = tgt_loader
        self.tgt_opt = tgt_opt
        self.device = device
        self.num_classes = num_classes
        self.args = args
        self.current_step = 0
        self.param_lr_c = []
        for param_group in self.tgt_opt.param_groups:
            self.param_lr_c.append(param_group["lr"])

    def lr_step(self):
        """
        Learning rate scheduler
        """
        if self.args.optimizer == "SGD":
            self.tgt_opt = utils.inv_lr_scheduler(
                self.param_lr_c, self.tgt_opt, self.current_step, init_lr=self.args.lr
            )

    def solve(self, epoch):
        pass


@register_solver("mmd")
class MMDSolver(BaseSolver):
    """
    Implements the MMD baseline from Contrastive Adaptation Network for Unsupervised DA: https://arxiv.org/abs/1901.00976
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_opt, device, num_classes, args):
        super(MMDSolver, self).__init__(
            net, src_loader, tgt_loader, tgt_opt, device, num_classes, args
        )
        # self.num_classes = 10

    def compute_paired_dist(self, A, B):
        bs_A = A.size(0)
        bs_T = B.size(0)
        feat_len = A.size(1)

        A_expand = A.unsqueeze(1).expand(bs_A, bs_T, feat_len)
        B_expand = B.unsqueeze(0).expand(bs_A, bs_T, feat_len)
        dists = (((A_expand - B_expand)) ** 2).sum(2)
        return dists

    def gamma_estimation(self, dist):
        dist_sum = (
            torch.sum(dist["ss"]) + torch.sum(dist["tt"]) + 2 * torch.sum(dist["st"])
        )
        bs_S = dist["ss"].size(0)
        bs_T = dist["tt"].size(0)
        N = bs_S * bs_S + bs_T * bs_T + 2 * bs_S * bs_T - bs_S - bs_T
        gamma = dist_sum.item() / N
        return gamma

    def compute_kernel_dist(self, dists, gamma, kernel_num=5, kernel_mul=2):
        base_gamma = gamma / (kernel_mul ** (kernel_num // 2))
        gamma_list = [base_gamma * (kernel_mul**i) for i in range(kernel_num)]
        gamma_tensor = (torch.tensor(gamma_list)).to(self.device)

        eps = 1e-5
        gamma_mask = (gamma_tensor < eps).type(torch.cuda.FloatTensor)
        gamma_tensor = (1.0 - gamma_mask) * gamma_tensor + gamma_mask * eps
        gamma_tensor = gamma_tensor.detach()

        dists = dists.unsqueeze(0) / gamma_tensor.view(-1, 1, 1)
        upper_mask = (dists > 1e5).type(torch.cuda.FloatTensor).detach()
        lower_mask = (dists < 1e-5).type(torch.cuda.FloatTensor).detach()
        normal_mask = 1.0 - upper_mask - lower_mask
        dists = normal_mask * dists + upper_mask * 1e5 + lower_mask * 1e-5
        kernel_val = torch.sum(torch.exp(-1.0 * dists), dim=0)
        return kernel_val

    def kernel_layer_aggregation(self, dist_layers, gamma_layers):
        kernel_dist = {}
        dists = dist_layers[0]
        gamma = gamma_layers[0]

        if len(kernel_dist.keys()) == 0:
            kernel_dist = {
                key: self.compute_kernel_dist(dists[key], gamma)
                for key in ["ss", "tt", "st"]
            }

        kernel_dist = {
            key: kernel_dist[key] + self.compute_kernel_dist(dists[key], gamma)
            for key in ["ss", "tt", "st"]
        }
        return kernel_dist

    def mmd(self, source, target):
        """
        Computes Maximum Mean Discrepancy
        """
        dists = {}
        dists["ss"] = self.compute_paired_dist(source, source)
        dists["tt"] = self.compute_paired_dist(target, target)
        dists["st"] = self.compute_paired_dist(source, target)

        # import pdb; pdb.set_trace();
        dist_layers, gamma_layers = [], []
        dist_layers += [dists]
        gamma_layers += [self.gamma_estimation(dists)]

        kernel_dist = self.kernel_layer_aggregation(dist_layers, gamma_layers)
        mmd = (
            torch.mean(kernel_dist["ss"])
            + torch.mean(kernel_dist["tt"])
            - 2.0 * torch.mean(kernel_dist["st"])
        )
        return mmd

    def solve(self, epoch):
        """
        Semisupervised adaptation via MMD
        """
        joint_loader = zip(self.src_loader, self.tgt_loader)

        src_sup_wt, lambda_unsup = self.args.lambda_src, self.args.lambda_unsup

        self.net.train()
        for (
            batch_idx,
            (
                ((_, data_s, _), label_s, _, (_, _, _)),
                ((_, data_t, _), _, _, (_, _, _)),
            ),
        ) in enumerate(joint_loader):
            self.current_step += 1
            self.tgt_opt.zero_grad()

            data_s, label_s = data_s.to(self.device), label_s.to(self.device)
            data_t = data_t.to(self.device)

            # Train with target labels
            score_s = self.net(data_s)
            xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s, label_s)

            info_str = "[Train MMD] Epoch: {}".format(epoch)
            info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())

            # extract and concat features

            score_t = self.net(data_t)
            # Kernel MMD
            mmd_loss = lambda_unsup * self.mmd(score_s, score_t)

            # Linear MMD
            # mmd_loss = lambda_unsup * torch.norm(
            # 	(score_s.mean(dim=0) - score_t.mean(dim=0)), 2
            # )

            loss = xeloss_src + mmd_loss
            loss.backward()

            self.tgt_opt.step()

            # Learning rate update
            self.lr_step()

            # log net update info
            info_str += " MMD loss: {:.3f}".format(mmd_loss.item())
            if batch_idx % 10 == 0:
                logger.info(info_str)


@register_solver("dann")
class DANNSolver(BaseSolver):
    """
    Implements DANN from Unsupervised Domain Adaptation by Backpropagation: https://arxiv.org/abs/1409.7495
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_opt, device, num_classes, args):
        super(DANNSolver, self).__init__(
            net, src_loader, tgt_loader, tgt_opt, device, num_classes, args
        )

    def solve(self, epoch, disc, disc_opt):
        """
        Unsupervised adaptation via DANN: https://arxiv.org/abs/1505.07818
        XE on labeled source + Domain Adversarial training on source+target
        """
        gan_criterion = nn.CrossEntropyLoss()

        self.net.train()
        disc.train()

        lambda_src, lambda_unsup = self.args.lambda_src, self.args.lambda_unsup

        joint_loader = zip(self.src_loader, self.tgt_loader)

        for (
            batch_idx,
            (
                ((_, data_s, _), label_s, _, (_, _, _)),
                ((_, data_t, _), _, _, (_, _, _)),
            ),
        ) in enumerate(joint_loader):
            self.current_step += 1
            self.tgt_opt.zero_grad()
            disc_opt.zero_grad()

            data_s, label_s = data_s.to(self.device), label_s.to(self.device)
            data_t = data_t.to(self.device)

            # Train with target labels
            score_s = self.net(data_s)
            xeloss_src = lambda_src * nn.CrossEntropyLoss()(score_s, label_s)

            info_str = "[Train DANN] Epoch: {}".format(epoch)
            info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())

            # extract and concat features
            score_t = self.net(data_t)
            f = torch.cat((score_s, score_t), 0)

            # predict with discriminator
            f_rev = utils.ReverseLayerF.apply(f)
            pred_concat = disc(f_rev)

            # import pdb; pdb.set_trace();
            pred_domain = torch.argmax(pred_concat, dim=1)
            target_dom_s = torch.ones(len(data_s)).long().to(self.device)
            target_dom_t = torch.zeros(len(data_t)).long().to(self.device)
            label_concat = torch.cat((target_dom_s, target_dom_t), 0)

            # import pdb; pdb.set_trace();
            domain_acc = (pred_domain == label_concat).sum() / len(label_concat)

            # compute loss for disciminator
            loss_domain = gan_criterion(pred_concat, label_concat)

            loss_final = (xeloss_src) + (lambda_unsup * loss_domain)
            loss_final.backward()

            # if domain_acc > 0.6:
            self.tgt_opt.step()
            disc_opt.step()

            # Learning rate update (if using SGD)
            self.lr_step()

            # log net update info
            info_str += " DANN loss: {:.3f}".format(lambda_unsup * loss_domain.item())
            info_str += " dom acc.: {:.1f}".format(domain_acc * 100)
            if batch_idx % 10 == 0:
                logger.info(info_str)


@register_solver("sentry")
class SENTRYSolver(BaseSolver):
    """
    Implements SENTRY
    """

    def __init__(self, net, src_loader, tgt_loader, tgt_opt, device, num_classes, args):
        super(SENTRYSolver, self).__init__(
            net, src_loader, tgt_loader, tgt_opt, device, num_classes, args
        )
        self.num_classes = args.num_classes
        self.queue_length = 256
        ra_obj = RandAugment(3, 2.0)
        self.tgt_loader.dataset.all_transforms = torchvision.transforms.Compose(
            [ra_obj, self.tgt_loader.dataset.base_transforms]
        )

    def compute_prf1(self, true_mask, pred_mask):
        """
        Compute precision, recall, and F1 metrics for predicted mask against ground truth
        """
        conf_mat = confusion_matrix(true_mask, pred_mask, labels=[False, True])
        p = conf_mat[1, 1] / (conf_mat[0, 1] + conf_mat[1, 1] + 1e-8)
        r = conf_mat[1, 1] / (conf_mat[1, 0] + conf_mat[1, 1] + 1e-8)
        f1 = (2 * p * r) / (p + r + 1e-8)
        return conf_mat, p, r, f1

    def solve(self, epoch):
        """
        Unsupervised Domain adaptation via SENTRY: Selective Entropy Optimization via Committee Consistency
        """

        joint_loader = zip(self.src_loader, self.tgt_loader)

        lambda_src, lambda_unsup, lambda_infoent = (
            self.args.lambda_src,
            self.args.lambda_unsup,
            self.args.lambda_infoent,
        )

        self.net.train()
        queue = torch.zeros(self.queue_length).to(self.device)
        pointer = 0
        for (
            batch_idx,
            (
                ((_, data_s, _), label_s, _, (_, _, _)),
                ((data_t_og, data_t, _), label_t, indices_t, (_, _, _)),
            ),
        ) in enumerate(joint_loader):
            self.current_step += 1
            data_s, label_s = data_s.to(self.device), label_s.to(self.device)
            data_t_og, data_t, label_t = (
                data_t_og.to(self.device),
                data_t.to(self.device),
                label_t.to(self.device),
            )

            # Train with target labels
            score_s = self.net(data_s)
            xeloss_src = lambda_src * nn.CrossEntropyLoss()(score_s, label_s)
            loss = xeloss_src

            info_str = "\n[Train SENTRY] Epoch: {}".format(epoch)
            info_str += " Source XE loss: {:.3f}".format(xeloss_src.item())

            score_t_og = self.net(data_t_og)
            batch_sz = data_t_og.shape[0]
            tgt_preds = score_t_og.max(dim=1)[1].reshape(-1)

            if (
                pointer + batch_sz > self.queue_length
            ):  # Deal with wrap around when ql % batchsize != 0
                rem_space = self.queue_length - pointer
                queue[pointer : self.queue_length] = tgt_preds[:rem_space].detach() + 1
                queue[0 : batch_sz - rem_space] = tgt_preds[rem_space:].detach() + 1
            else:
                queue[pointer : pointer + batch_sz] = tgt_preds.detach() + 1
            pointer = (pointer + batch_sz) % self.queue_length

            bincounts = (
                torch.bincount(queue.long(), minlength=self.num_classes + 1).float()
                / self.queue_length
            )
            bincounts = bincounts[1:]

            log_q = torch.log(bincounts + 1e-12).detach()
            loss_infoent = lambda_infoent * torch.mean(
                torch.sum(
                    score_t_og.softmax(dim=1) * log_q.reshape(1, self.num_classes),
                    dim=1,
                )
            )
            loss += loss_infoent
            info_str += " Infoent loss: {:.3f}".format(loss_infoent.item())

            score_t_og = self.net(data_t_og).detach()
            tgt_preds = score_t_og.max(dim=1)[1].reshape(-1)

            # When pseudobalancing, label_t will correspond to pseudolabels rather than ground truth, so use backup instead
            if self.args.pseudo_balance_target:
                label_t = torch.from_numpy(
                    np.array(self.tgt_loader.dataset.actual_targets)
                )[indices_t].to(self.device)

            # Compute actual correctness mask for analysis only
            correct_mask_gt = tgt_preds.detach().cpu() == label_t.cpu()
            score_t_aug = self.net(data_t.to(self.device))
            tgt_preds_aug = score_t_aug.max(dim=1)[1].reshape(-1)
            correct_mask = (tgt_preds == tgt_preds_aug).detach()
            incorrect_mask = (tgt_preds != tgt_preds_aug).detach()

            # Compute some stats
            correct_ratio = (correct_mask).sum().item() / data_t_og.shape[0]
            incorrect_ratio = (incorrect_mask).sum().item() / data_t_og.shape[0]
            (
                _,
                correct_precision,
                _,
                _,
            ) = self.compute_prf1(
                correct_mask_gt.cpu().numpy(), correct_mask.cpu().numpy()
            )
            info_str += (
                "\n {:d} / {:d} consistent ({:.2f}): GT precision: {:.2f}".format(
                    correct_mask.sum(),
                    data_t_og.shape[0],
                    correct_ratio,
                    correct_precision,
                )
            )

            if correct_ratio > 0.0:
                probs_t_pos = F.softmax(score_t_aug, dim=1)
                loss_cent_correct = (
                    lambda_unsup
                    * correct_ratio
                    * -torch.mean(
                        torch.sum(
                            probs_t_pos[correct_mask]
                            * (torch.log(probs_t_pos[correct_mask] + 1e-12)),
                            1,
                        )
                    )
                )
                loss += loss_cent_correct
                info_str += " SENTRY loss (consistent): {:.3f}".format(
                    loss_cent_correct.item()
                )

            if incorrect_ratio > 0.0:
                probs_t_neg = F.softmax(score_t_aug, dim=1)
                loss_cent_incorrect = (
                    lambda_unsup
                    * incorrect_ratio
                    * torch.mean(
                        torch.sum(
                            probs_t_neg[incorrect_mask]
                            * (torch.log(probs_t_neg[incorrect_mask] + 1e-12)),
                            1,
                        )
                    )
                )
                loss += loss_cent_incorrect
                info_str += " SENTRY loss (inconsistent): {:.3f}".format(
                    loss_cent_incorrect.item()
                )

            # Backprop
            self.tgt_opt.zero_grad()
            loss.backward()
            self.tgt_opt.step()

            # Learning rate update (if using SGD)
            self.lr_step()

            if batch_idx % 10 == 0:
                logger.info(info_str)

# Domain Adaptation experiments
import os
import sys
import random
import argparse
import copy
import pprint
import distutils
import distutils.util
from omegaconf import OmegaConf
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt
from loguru import logger
import wandb

import numpy as np

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from apex.parallel import DistributedDataParallel as ApexDDP
import distributed as dist

from adapt.solver import get_solver
from model import ResNet50
from data import UDADataset
import utils
from adapt import *


os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
# Load existing configuration?
parser.add_argument(
    "--load_from_cfg",
    type=lambda x: bool(distutils.util.strtobool(x)),
    default=True,
    help="Load from config?",
)
parser.add_argument(
    "--cfg_file",
    type=str,
    help="Experiment configuration file",
    default="config/dollarstreet/lmda.yml",
)
# Experiment identifer
parser.add_argument("--id", type=str, help="Experiment identifier")
parser.add_argument("--use_cuda", help="Use GPU?")
parser.add_argument("--num_runs", type=int, help="Number of runs")
# Source and target domain
parser.add_argument("--benchmark", help="Adaptation benchmark")
parser.add_argument("--source", help="Source dataset")
parser.add_argument("--target", help="Target dataset")
# CNN parameters
parser.add_argument("--cnn", type=str, help="CNN architecture")
parser.add_argument(
    "--wandb",
    type=lambda x: bool(distutils.util.strtobool(x)),
    default=True,
    help="Load source checkpoint?",
)
parser.add_argument(
    "--load_source",
    type=lambda x: bool(distutils.util.strtobool(x)),
    default=True,
    help="Load source checkpoint?",
)
parser.add_argument("--source_id", type=str, help="Identifier for source checkpoint")
parser.add_argument(
    "--resume_source_training",
    type=lambda x: bool(distutils.util.strtobool(x)),
    default=False,
    help="Resume source training from checkpoint (if exists)",
)

parser.add_argument(
    "--l2_normalize",
    type=lambda x: bool(distutils.util.strtobool(x)),
    help="L2 normalize features?",
)
parser.add_argument("--temperature", type=float, help="CNN softmax temperature")
# Class balancing parameters
parser.add_argument(
    "--class_balance_source",
    type=lambda x: bool(distutils.util.strtobool(x)),
    help="Class-balance source?",
)
parser.add_argument(
    "--pseudo_balance_target",
    type=lambda x: bool(distutils.util.strtobool(x)),
    help="Pseudo class-balance target?",
)
parser.add_argument(
    "--align_dataloaders",
    type=lambda x: bool(distutils.util.strtobool(x)),
    help="Align source and target dataloaders?",
)
parser.add_argument(
    "--class_conditioned_thresholding",
    type=lambda x: bool(distutils.util.strtobool(x)),
    help="Class conditioned thresholding?",
)
# DA details
parser.add_argument("--da_strat", type=str, help="DA strategy")
parser.add_argument(
    "--load_da",
    type=lambda x: bool(distutils.util.strtobool(x)),
    help="Load saved DA checkpoint?",
)
parser.add_argument(
    "--resume_da",
    type=lambda x: bool(distutils.util.strtobool(x)),
    default=False,
    help="Resume DA adaptation from checkpoint (if exists)",
)
# Training details
parser.add_argument("--optimizer", type=str, help="Optimizer")
parser.add_argument("--batch_size", type=int, help="Batch size")
parser.add_argument("--lr", type=float, help="Learning rate")
parser.add_argument("--wd", type=float, help="Weight decay")
parser.add_argument("--num_epochs", type=int, help="Number of Epochs")
parser.add_argument("--da_lr", type=float, help="Unsupervised DA Learning rate")
parser.add_argument("--da_num_epochs", type=int, help="DA Number of epochs")
# Loss weights
parser.add_argument("--lambda_src", type=float, help="Source supervised XE loss weight")
parser.add_argument(
    "--confidence_threshold", type=float, help="Source supervised XE loss weight"
)
parser.add_argument("--lambda_att", type=float, help="Attention loss weight")
parser.add_argument(
    "--lambda_unsup", type=float, help="Target unsupervised loss weight"
)
parser.add_argument("--lambda_infoent", type=float, help="Target infoent loss weight")
parser.add_argument(
    "--use_lm",
    type=lambda x: bool(distutils.util.strtobool(x)),
    help="Load source checkpoint?",
)
parser.add_argument(
    "--use_wiki",
    type=lambda x: bool(distutils.util.strtobool(x)),
    help="Load source checkpoint?",
)
# Distributed training arguments.
parser.add_argument(
    "--num-machines",
    type=int,
    default=1,
    help="Number of machines used in distributed training.",
)
parser.add_argument(
    "--num-gpus-per-machine",
    type=int,
    default=1,
    help="""Number of GPUs per machine with IDs as (0, 1, 2 ...). Set as
	zero for single-process CPU training.""",
)
parser.add_argument(
    "--machine-rank",
    type=int,
    default=0,
    help="""Rank of the machine, integer in [0, num_machines). Default 0
	for training with a single machine.""",
)
parser.add_argument(
    "--dist-url",
    default="tcp://127.0.0.1:8898",
    type=str,
    help="url used to set up distributed training",
)
parser.add_argument(
    "--dist-backend", default="nccl", type=str, help="distributed backend"
)
parser.add_argument(
    "--seed", default=1234, type=int, help="seed for initializing training. "
)


def main(args: argparse.Namespace):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    device = torch.cuda.current_device()
    if args.run == 1:
        logger.remove(0)

    exp_name = "{:s}_{:s}_{}_{}_net_{:s}_{:s}".format(
        args.id, args.da_strat, args.da_lr, args.cnn, args.source, args.target
    )

    # Add a logger for stdout only for the master process.
    if dist.is_master_process():
        logger.add(
            sys.stdout, format="<g>{time}</g>: <lvl>{message}</lvl>", colorize=True
        )

    if dist.get_world_size() > 1:
        logger.info(
            f"Current process: Rank {dist.get_rank()}, World size {dist.get_world_size()}"
        )

    logger.info("World size: {}".format(dist.get_world_size()))

    args.batch_size = int(args.batch_size / args.num_gpus_per_machine)
    ##############################################################################################################
    # Setup source data loaders
    ##############################################################################################################
    logger.info("Loading {} dataset".format(args.source))
    source_data_obj = UDADataset(
        args.benchmark,
        args.source,
        is_target=False,
        dist=(dist.get_world_size() > 1),
    )
    source_dataset = source_data_obj.get_dataset()
    (
        source_train_loader,
        source_test_loader,
        src_train_idx,
        src_test_idx,
    ) = source_data_obj.get_loaders(
        test_ratio=0.2,
        class_balance_train=args.class_balance_source,
        batch_size=args.batch_size,
        # subsample_size=2637  # len(src_train_idx)  # subsample
    )

    # import pdb
    # pdb.set_trace()

    num_classes = source_data_obj.get_num_classes()
    args.num_classes = num_classes
    logger.info("Number of classes: {}".format(num_classes))

    ##############################################################################################################
    # Train / load a source model
    ##############################################################################################################
    source_model = ResNet50(
        num_cls=num_classes,
        l2_normalize=args.l2_normalize,
        temperature=args.temperature,
        pretrained=True,
    ).to(device)

    cudnn.benchmark = True

    source_load_id = args.source_id if args.source_id else args.id
    source_file = "{}_{}_source_{}_lm{}.pth".format(
        args.source, args.cnn, source_load_id, args.use_lm
    )
    # source_file = "{}_{}_source_lm{}.pth".format(args.source, args.cnn, args.use_lm)

    source_path = os.path.join("checkpoints", "source", source_file)

    if args.use_wiki:
        assert args.use_lm, "Can only use wiki if use_lm == True"

    if args.load_source and os.path.exists(source_path):
        logger.info("Found source checkpoint at {}".format(source_path))
    else:
        if args.resume_source_training and os.path.exists(source_path):
            logger.info(
                "Found source checkpoint at {}, resuming training...".format(
                    source_path
                )
            )
            source_model.load_state_dict(torch.load(source_path))
        else:
            if not os.path.exists(source_path):
                logger.info("Source checkpoint not found at at {}".format(source_path))
            logger.info("Training source checkpoint...")

        # Wrap model in ApexDDP if using more than one processes.
        if dist.get_world_size() > 1:
            dist.synchronize()
            source_model = ApexDDP(source_model, delay_allreduce=True)

        utils.train_source_model(
            source_model,
            source_train_loader,
            source_test_loader,
            num_classes,
            args,
            device,
        )

    model = source_model
    model.load_state_dict(torch.load(source_path), strict=False)

    out_str = ""
    if dist.is_master_process():
        acc_source_train, cm_source_train, brec_dict_source_train = utils.test(
            model,
            device,
            source_train_loader,
            split="train",
            num_classes=num_classes,
        )

        per_class_acc_source_train = cm_source_train.diagonal().numpy() / (
            cm_source_train.sum(axis=1).numpy() + 1e-8
        )
        per_class_acc_source_train = per_class_acc_source_train.mean() * 100
        br5_source_train = (
            np.array(list(brec_dict_source_train.values())).mean() * 100.0
        )

        acc_source, cm_source, brec_dict_source_test = utils.test(
            model,
            device,
            source_test_loader,
            split="test",
            num_classes=num_classes,
        )

        per_class_acc_source = cm_source.diagonal().numpy() / (
            cm_source.sum(axis=1).numpy() + 1e-8
        )
        per_class_acc_source = per_class_acc_source.mean() * 100
        br5_source_test = np.array(list(brec_dict_source_test.values())).mean() * 100.0

        out_str += "Source performance on heldout {}:\t br@1/br@5={:.2f}%/{:.2f}% (train={:.2f}%/{:.2f}%) \
				\tAgg. acc={:.2f}% (train={:.2f}%)".format(
            args.source,
            per_class_acc_source,
            br5_source_test,
            per_class_acc_source_train,
            br5_source_train,
            acc_source,
            acc_source_train,
        )
        logger.info(out_str)

    ##############################################################################################################
    # Setup target data loaders
    ##############################################################################################################
    logger.info("\nLoading {} dataset".format(args.target))

    target_data_obj = UDADataset(
        args.benchmark,
        args.target,
        is_target=True,
        dist=(dist.get_world_size() > 1),
    )
    target_dataset = target_data_obj.get_dataset()

    # target_test_ratio = 0.0 if args.benchmark == "dollarstreet" else 0.2
    # target_test_ratio = 0.0
    target_test_ratio = 0.2

    (
        target_train_loader,
        target_test_loader,
        tgt_train_idx,
        tgt_test_idx,
    ) = target_data_obj.get_loaders(
        test_ratio=target_test_ratio,
        class_balance_train=False,
        batch_size=args.batch_size,
    )
    # import pdb
    # pdb.set_trace()

    perf_dict = {}
    if dist.is_master_process():
        acc_before, cm_before, brec_dict_before = utils.test(
            model, device, target_test_loader, split="test", num_classes=num_classes
        )
        per_class_acc_before = cm_before.diagonal().numpy() / (
            cm_before.sum(axis=1).numpy() + 1e-8
        )
        per_class_acc_before = per_class_acc_before.mean() * 100
        br5_before = np.array(list(brec_dict_before.values())).mean() * 100.0

        out_str = "{}->{}, Before {}:\t Avg. br1/br5={:.2f}%/{:.2f}%\tAgg. acc={:.2f}%".format(
            args.source,
            args.target,
            args.da_strat,
            per_class_acc_before,
            br5_before,
            acc_before,
        )
        logger.info(out_str)

        perf_dict["agg_acc_src"] = acc_source
        perf_dict["avg_acc_src"] = per_class_acc_source
        perf_dict["avg_br5_src"] = br5_source_test

        perf_dict["agg_acc_before"] = acc_before
        perf_dict["avg_acc_before"] = per_class_acc_before
        perf_dict["avg_br5_before"] = br5_before

        wargs = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
        perf_dict["args"] = wargs
        utils.log(perf_dict, exp_name, args)

    # return
    # Wrap in Apex DDP
    if dist.get_world_size() > 1:
        dist.synchronize()
        model = ApexDDP(model, delay_allreduce=True)

    # return
    # sys.exit(0)
    ################################################################################################################
    # Unsupervised adaptation of source model to target
    ################################################################################################################

    if dist.is_master_process():
        if "r1" in args.id:
            wandb.init(project="inclusive-da", entity="virajp", name=exp_name)
            wandb.config.update(args)
            wandb.log({"target_acc": per_class_acc_before}, step=0)

    outdir = "checkpoints"
    os.makedirs(os.path.join(outdir, args.da_strat), exist_ok=True)
    outfile = os.path.join(outdir, args.da_strat, "{}.pth".format(exp_name))

    args.load_da = True
    if args.load_da and os.path.exists(outfile):
        logger.info(
            "Trained {} checkpoint found: {}, loading...\n".format(
                args.da_strat, outfile
            )
        )
        source_model_adapt = ResNet50(
            num_cls=num_classes,
            l2_normalize=args.l2_normalize,
            temperature=args.temperature,
        )
        source_model_adapt.load(outfile)
        source_model_adapt.to(device)
    else:
        if args.resume_da and os.path.exists(outfile):
            logger.info(
                "Trained {} checkpoint found: {}, resuming DA...\n".format(
                    args.da_strat, outfile
                )
            )
            source_model_adapt = ResNet50(
                num_cls=num_classes,
                l2_normalize=args.l2_normalize,
                temperature=args.temperature,
            )
            source_model_adapt.load(outfile)
        else:
            source_model_adapt = model
            logger.info(
                "Training {} {} model for {}->{} \n".format(
                    args.da_strat, args.cnn, args.source, args.target
                )
            )
        source_model_adapt.to(device)

        opt_net = utils.generate_optimizer(source_model_adapt, args, mode="da")

        solver = get_solver(
            args.da_strat,
            source_model_adapt,
            source_train_loader,
            target_train_loader,
            opt_net,
            device,
            num_classes,
            args,
        )

        # import pdb; pdb.set_trace();
        for epoch in range(args.da_num_epochs):
            if args.pseudo_balance_target:
                # raise NotImplementedError
                if dist.is_master_process():
                    logger.info(
                        "\nEpoch {}: Re-estimating probabilities for pseudo-balancing...".format(
                            epoch
                        )
                    )
                # Approximately class-balance target dataloader using pseudolabels at the start of each epoch
                target_data_obj_copy = copy.deepcopy(target_data_obj)

                _, _, plabels, img_ids_tgt = utils.get_embedding(
                    solver.net, target_train_loader, device, num_classes
                )

                target_data_obj_copy.dataset.actual_targets = copy.deepcopy(
                    target_data_obj_copy.dataset.targets
                )  # Create backup of actual labels
                target_data_obj_copy.dataset.targets = plabels.numpy()
            
                (
                    tgt_train_loader_pbalanced,
                    _,
                    _,
                    _,
                ) = target_data_obj_copy.get_loaders(
                    test_ratio=target_test_ratio,
                    class_balance_train=True,
                    batch_size=args.batch_size,
                )
                tgt_train_loader_pbalanced.dataset.actual_targets = (
                    target_data_obj_copy.dataset.actual_targets
                )
                solver.tgt_loader = tgt_train_loader_pbalanced

            if args.da_strat == "dann":
                discriminator = utils.get_discriminator(num_classes).to(device)
                opt_dis = optim.SGD(
                    discriminator.parameters(),
                    lr=args.da_lr,
                    weight_decay=args.da_wd,
                    momentum=0.9,
                    nesterov=True,
                )
                solver.solve(epoch, discriminator, opt_dis)
            else:
                solver.solve(epoch)

            if ("r1" in args.id) and args.wandb:
                logger.info(
                    "\nEvaluating adapted model on {} test set at epoch {}".format(
                        args.target, epoch
                    )
                )
                _, cm_after_epoch, _ = utils.test(
                    source_model_adapt,
                    device,
                    target_test_loader,
                    split="test",
                    num_classes=num_classes,
                )
                per_class_acc_after_epoch = (
                    cm_after_epoch.diagonal().numpy()
                    / cm_after_epoch.sum(axis=1).numpy()
                )
                per_class_acc_after_epoch = per_class_acc_after_epoch.mean() * 100
                wandb.log({"target_acc": per_class_acc_after_epoch}, step=epoch)

                logger.info("###################################")
                print_str = "\n\t\t\tAfter epoch{}:\t br@1: {:.2f}%".format(
                    epoch, per_class_acc_after_epoch
                )
                logger.info(print_str)

        if dist.is_master_process():
            logger.info("Saving to", outfile)
            source_model_adapt = solver.net
            save_model = (
                source_model_adapt.module
                if dist.get_world_size() > 1
                else source_model_adapt
            )
            torch.save(save_model.state_dict(), outfile)

            if "r1" in args.id and args.wandb and args.da_strat == "lmda":
                norm_gt_ratio = (
                    (
                        (solver.per_class_count_gt * 100.0)
                        / solver.per_class_count_gt.sum()
                    )
                    .cpu()
                    .numpy()
                )

                fig = plt.figure(figsize=(9, 6))
                ix2cname = {v: k for k, v in target_data_obj.cname2ix.items()}
                plt.bar([ix2cname[ix] for ix in np.arange(num_classes)], norm_gt_ratio)
                plt.xlabel("Class")
                plt.xticks(rotation=90)
                plt.xlabel("%")
                wandb.log({"tgt_seen_hist": wandb.Image(fig)})

                norm_gt_sel_ratio = (
                    (
                        (solver.per_class_count_sel_gt * 100.0)
                        / solver.per_class_count_sel_gt.sum()
                    )
                    .cpu()
                    .numpy()
                )

                fig = plt.figure(figsize=(9, 6))
                ix2cname = {v: k for k, v in target_data_obj.cname2ix.items()}
                plt.bar(
                    [ix2cname[ix] for ix in np.arange(num_classes)], norm_gt_sel_ratio
                )
                plt.xlabel("Class")
                plt.xticks(rotation=90)
                plt.xlabel("%")
                wandb.log({"tgt_sel_hist": wandb.Image(fig)})

    if dist.is_master_process():
        # Evaluate adapted model
        logger.info("\nEvaluating adapted model on {} test set".format(args.target))
        acc_after, cm_after, brec_dict_after = utils.test(
            source_model_adapt,
            device,
            target_test_loader,
            split="test",
            num_classes=num_classes,
        )
        per_class_acc_after = cm_after.diagonal().numpy() / cm_after.sum(axis=1).numpy()
        per_class_acc_after = per_class_acc_after.mean() * 100
        br5_after = np.array(list(brec_dict_after.values())).mean() * 100.0

        logger.info("###################################")
        out_str += (
            "\n\t\t\tAfter {}:\t Avg. br1/br5={:.2f}/{:.2f}%\tAgg. acc={:.2f}%".format(
                args.da_strat, per_class_acc_after, br5_after, acc_after
            )
        )
        logger.info(out_str)

        perf_dict["agg_acc_after"] = acc_after
        perf_dict["avg_acc_after"] = per_class_acc_after
        perf_dict["avg_br5_after"] = br5_after
        wargs = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
        perf_dict["args"] = wargs
        utils.log(perf_dict, exp_name, args)

        # if args.benchmark == 'dollarstreet':
        utils.plot_accuracy_statistics(
            cm_before,
            cm_after,
            num_classes,
            args,
            target_train_loader,
            target_data_obj.cname2ix,
            exp_name,
        )


if __name__ == "__main__":
    args_cmd = parser.parse_args()
    if args_cmd.load_from_cfg:
        args_cfg = dict(OmegaConf.load(args_cmd.cfg_file))
        args_cmd = vars(args_cmd)
        for k in args_cmd.keys():
            if args_cmd[k] is not None:
                args_cfg[k] = args_cmd[k]
        _A = OmegaConf.create(args_cfg)
    else:
        _A = args_cmd

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(_A)

    seeds = [1234, 5678, 42]
    original_id, original_source_id = _A.id, _A.source_id
    for run in range(1, _A.num_runs + 1):
        _A.seed = seeds[run - 1]
        _A.id = "{}_r{}".format(original_id, run)
        _A.source_id = "{}_r{}".format(original_source_id, run)
        _A.run = run
        if _A.num_gpus_per_machine == 0:
            raise NotImplementedError("Training on CPU is not supported.")
        else:
            # This will launch `main` and set appropriate CUDA device (GPU ID) as
            # per process (accessed in the beginning of `main`).
            if _A.num_gpus_per_machine == 1:
                main(_A)
            else:
                dist.launch(
                    main,
                    num_machines=_A.num_machines,
                    num_gpus_per_machine=_A.num_gpus_per_machine,
                    machine_rank=_A.machine_rank,
                    dist_url=_A.dist_url,
                    dist_backend=_A.dist_backend,
                    args=(_A,),
                )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import random
import os
import argparse
import logging
import numpy as np
import torch
from model import resnet20
from clients import Clients
from utils import load_dataset
from fl import run_fl


def arg_parse():
    parser = argparse.ArgumentParser()

    # Dataset part
    parser.add_argument("--dataset", type=str, default="CIFAR10",
                        help="dataset, possible are `EMNIST`, `FashionMNIST`,"
                             "`CIFAR10`, `CIFAR100`")
    parser.add_argument("--n_clients", type=int, default=10)
    parser.add_argument(
        "--train_frac", help="fraction of training samples", type=float,
        default=0.9)
    parser.add_argument(
        "--valid_frac", help="fraction of validation samples in train samples",
        type=float, default=0.1)
    parser.add_argument("--pathological_split", help="if selected, the dataset"
                        "will be split as in 'Communication-Efficient Learning"
                        "of Deep Networks from Decentralized Data'; i.e., each"
                        "client will receive `n_shards` of dataset, where each"
                        "shard contains at most two classes",
                        action="store_true"
                        )
    parser.add_argument("--n_shards", help="number of shards given to each"
                        "client; ignored if `--pathological_split` is not"
                        "used; default is 2", type=int, default=2)
    parser.add_argument(
        "--alpha", help="the parameter of dirichalet", type=float, default=0.1)

    # Training part
    parser.add_argument("--lr", type=float, default=0.1, help="applies to SGD")
    parser.add_argument("--batch_size", type=int,
                        default=128, help="training batch size")
    parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
    parser.add_argument("--gpu", type=str, default="0", help="use of gpu")
    parser.add_argument("--log_dir", type=str,
                        default="log", help="directory of logs")
    parser.add_argument("--frac", type=float, default=1,
                        help="fraction of participating clients")
    parser.add_argument("--rounds", type=int,
                        default=200, help="number of training rounds")
    parser.add_argument("--local_epochs", type=int, default=1,
                        help="number of local training epochs")
    parser.add_argument("--eval_interval", type=int,
                        default=1, help="interval of evalution")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--fed_method", type=str,
                        default="FedAvg", help="`FedAvg`, `FedProx` or `Ditto`")
    parser.add_argument("--mu", type=float, default=0.01,
                        help="constant for proximal in `FedProx` method")
    parser.add_argument("--lam", type=float, default=0.1,
                        help="constant for proximal in `Ditto` method")

    args = parser.parse_args()
    return args


def seed_everything(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def init_logger(args):
    log_file = os.path.join(args.log_dir, args.dataset + ".log")

    logging.basicConfig(
        format="%(asctime)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode="w+"
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def main():
    args = arg_parse()

    seed_everything(args)

    init_logger(args)

    client_train_datasets, client_valid_datasets, client_test_datasets, \
        data_info = load_dataset(args)

    clients = Clients(lambda: resnet20(in_channels=data_info["num_channels"],
                                       num_classes=data_info["num_classes"]),
                      lambda x: torch.optim.SGD(x, lr=args.lr, momentum=0.9),
                      args, client_train_datasets, client_valid_datasets,
                      client_test_datasets)

    run_fl(clients, args)


if __name__ == "__main__":
    main()

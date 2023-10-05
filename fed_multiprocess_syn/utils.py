# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Normalize
from subset import CustomSubset


def load_dataset(args):
    sys.path.append(os.path.dirname(__file__) + os.sep + "../")
    from data_utils.data_split import split_noniid, \
        pathological_non_iid_split
    from data_utils.plot import display_data_distribution

    if args.dataset == "EMNIST":
        transform = transforms.Compose([ToTensor()])
        # train = True, create data from training dataset
        train_data = datasets.EMNIST(
            root="./data", split="byclass", download=True, transform=transform,
            train=True)
        # test = False, create data from test dataset
        test_data = datasets.EMNIST(
            root="./data", split="byclass", download=True, transform=transform,
            train=False)
    elif args.dataset == "FashionMNIST":
        transform = transforms.Compose([ToTensor()])
        # train = True, create data from training dataset
        train_data = datasets.FashionMNIST(
            root="./data", download=True, transform=transform, train=True)
        # test = False, create data from test dataset
        test_data = datasets.FashionMNIST(
            root="./data", download=True, transform=transform, train=False)
    elif args.dataset == "CIFAR10":
        transform = transforms.Compose([
            ToTensor(),
            # Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        # train = True, create data from training dataset
        train_data = datasets.CIFAR10(
            root="./data", download=True, transform=transform, train=True)
        # test = False, create data from test dataset
        test_data = datasets.CIFAR10(
            root="./data", download=True, transform=transform, train=False)
    elif args.dataset == "CIFAR100":
        transform = transforms.Compose([ToTensor()])
        # train = True, create data from training dataset
        train_data = datasets.CIFAR100(
            root="./data", download=True, transform=transform, train=True)
        # test = False, create data from test dataset
        test_data = datasets.CIFAR100(
            root="./data", download=True, transform=transform, train=False)
    else:
        raise ValueError("Please input the correct dataset name, it must be"
                         "one of: EMNIST, FashionMNIST, CIFAR10, CIFAR100!")

    data_info = {}
    data_info["classes"] = train_data.classes
    data_info["num_classes"] = len(train_data.classes)
    data_info["input_size"] = train_data.data[0].shape[0]
    if len(train_data.data[0].shape) == 2:
        data_info["num_channels"] = 1
    else:
        data_info["num_channels"] = train_data.data[0].shape[-1]

    labels = np.concatenate(
        [np.array(train_data.targets), np.array(test_data.targets)], axis=0)
    dataset = ConcatDataset([train_data, test_data])

    if args.pathological_split:
        client_idcs = pathological_non_iid_split(
            labels, args.n_shards, args.n_clients)
    else:
        client_idcs = split_noniid(labels, args.alpha, args.n_clients)

    display_data_distribution(
        client_idcs, labels, data_info["num_classes"], args.n_clients, args)

    client_train_idcs, client_test_idcs, client_valid_idcs = [], [], []
    # Before dividing the local training, validation, and test datasets,
    # shuffle must be performed first
    for idcs in client_idcs:
        train_idcs, test_idcs =\
            train_test_split(
                idcs,
                train_size=args.train_frac,
                random_state=args.seed
            )
        if args.valid_frac > 0:
            train_idcs, valid_idcs = \
                train_test_split(
                    train_idcs,
                    train_size=1.-args.valid_frac,
                    random_state=args.seed
                )
            client_valid_idcs.append(valid_idcs)
        else:
            client_valid_idcs.append([])
        client_train_idcs.append(train_idcs)
        client_test_idcs.append(test_idcs)

    client_train_datasets = [CustomSubset(
        dataset, idcs) for idcs in client_train_idcs]
    client_valid_datasets = [CustomSubset(
        dataset, idcs) for idcs in client_valid_idcs]
    client_test_datasets = [CustomSubset(
        dataset, idcs) for idcs in client_test_idcs]

    return client_train_datasets, client_valid_datasets, \
        client_test_datasets, data_info


def init_clients_weights(clients):
    """Initialize the aggretation weight.
    """
    client_n_samples_train = [client.n_samples_train for client in clients]
    client_n_samples_valid = [client.n_samples_valid for client in clients]
    client_n_samples_test = [client.n_samples_test for client in clients]

    samples_sum_train = np.sum(client_n_samples_train)
    samples_sum_valid = np.sum(client_n_samples_valid)
    samples_sum_test = np.sum(client_n_samples_test)
    for client in clients:
        client.train_weight = client.n_samples_train/samples_sum_train
        client.valid_weight = client.n_samples_valid/samples_sum_valid
        client.test_weight = client.n_samples_test/samples_sum_test

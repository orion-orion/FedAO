# -*- coding: utf-8 -*-
import random
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_dataset(args):
    sys.path.append(os.path.dirname(__file__) + os.sep + "../")
    from data_utils.data_split import split_noniid, \
        pathological_non_iid_split
    from data_utils.plot import display_data_distribution

    if args.dataset == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10\
            .load_data()
    elif args.dataset == "CIFAR100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100\
            .load_data()
    else:
        raise ValueError("Please input the correct dataset name, it must be"
                         "one of: CIFAR10, CIFAR100!")

    X = np.concatenate([x_train, x_test], axis=0)
    X = X.astype(np.float32) / 255.0  # [0 - 1] range
    y = np.concatenate([y_train, y_test], axis=0).reshape(-1)

    data_info = {}
    data_info["classes"] = list(set(y.tolist()))
    data_info["num_classes"] = y.max()+1
    data_info["num_features"] = X[0].shape[0]
    data_info["num_channels"] = X[0].shape[-1]

    if args.pathological_split:
        client_idcs = pathological_non_iid_split(
            y, args.n_shards, args.n_clients)
    else:
        client_idcs = split_noniid(y, args.alpha, args.n_clients)

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

    display_data_distribution(
        client_idcs, y, data_info["num_classes"], args.n_clients, args)

    client_train_datasets = [(X[train_idc], y[train_idc])
                             for train_idc in client_train_idcs]
    client_valid_datasets = [(X[valid_idc], y[valid_idc])
                             for valid_idc in client_valid_idcs]
    client_test_datasets = [(X[test_idc], y[test_idc])
                            for test_idc in client_test_idcs]

    return client_train_datasets, client_valid_datasets, \
        client_test_datasets, data_info


def batch_iter(dataset, batch_size, mode="train"):
    X, Y = dataset
    x_y_pair = list(zip(X, Y))
    if mode == "train":
        random.shuffle(x_y_pair)
    X = np.stack(list(zip(*x_y_pair))[0])
    Y = np.stack(list(zip(*x_y_pair))[1])

    if len(dataset[0]) % batch_size == 0:
        n_batch = len(dataset[0])//batch_size
    else:
        n_batch = len(dataset[0])//batch_size + 1
    for batch_i in range(0, n_batch):
        start_i = batch_i * batch_size
        x = X[start_i: start_i + batch_size]
        y = Y[start_i: start_i + batch_size]
        yield (x, y)

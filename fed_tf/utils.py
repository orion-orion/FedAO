import html
import re
import nltk
import os
import pickle
import random
import numpy as np
# from torchvision import datasets, transforms
from nltk import word_tokenize, pos_tag
import tensorflow as tf
import sys


sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from data_utils.data_split import split_noniid, pathological_non_iid_split
from data_utils.plot import display_data_distribution

def load_dataset(args):
    if args.dataset == "CIFAR10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif args.dataset == "CIFAR100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    else:
        raise ValueError("Invalid dataset!")
    
    X = np.concatenate([x_train, x_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0).reshape(-1, )
    data_info = {}
    data_info["classes"] = list(set(y.tolist()))
    data_info["num_classes"] = y.max()+1
    data_info["num_features"] = X[0].shape[0]
    data_info["num_channels"] = X[0].shape[-1]

    if args.pathological_split:
        client_idcs = pathological_non_iid_split(y.reshape(-1, ), args.n_shards, args.n_clients)
    else:
        client_idcs = split_noniid(y.reshape(-1, ), args.alpha, args.n_clients)


    client_train_idcs, client_test_idcs, client_val_idcs = [], [], []
    for idcs in client_idcs:
        n_samples = len(idcs)
        n_train = int(n_samples * args.train_frac)
        n_test = n_samples - n_train
        if args.val_frac > 0:
            n_val = int(n_train * args.val_frac)
            n_train = n_train - n_val
            client_val_idcs.append(idcs[n_train:(n_train+n_val)])
        else:
            client_val_idcs.append([])
        client_train_idcs.append(idcs[:n_train])
        client_test_idcs.append(idcs[n_test:])

    display_data_distribution(client_idcs, y, data_info['num_classes'], args.n_clients, args)

    client_train_datasets = [(X[train_idc], y[train_idc]) for train_idc in client_train_idcs]
    client_valid_datasets = [(X[train_idc], y[train_idc]) for train_idc in client_train_idcs]
    client_test_datasets = [(X[train_idc], y[train_idc]) for train_idc in client_train_idcs]

    return client_train_datasets, client_valid_datasets, client_test_datasets, data_info


def batch_iter(dataset, batch_size):
    X, Y = dataset
    x_y_pair = [ (x, y) for (x, y) in zip(X, Y)] 
    random.shuffle(x_y_pair)
    X = np.stack(list(zip(*x_y_pair))[0])
    Y = np.stack(list(zip(*x_y_pair))[1])

    if len(dataset) % batch_size == 0:
        n_batch = len(dataset)//batch_size
    else:
        n_batch = len(dataset)//batch_size + 1
    for batch_i in range(0, n_batch):
        start_i = batch_i * batch_size
        x = X[start_i: start_i + batch_size]
        y = Y[start_i: start_i + batch_size]
        yield (x, y)
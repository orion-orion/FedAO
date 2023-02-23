# -*- coding: utf-8 -*-
import random
import os
import argparse
import numpy as np
import tensorflow as tf
from scipy import sparse
import logging
from utils import load_dataset
from model import ConvNet
from clients import Clients
from fl import run_fl


def arg_parse():
    parser = argparse.ArgumentParser()

    # dataset part
    parser.add_argument('--dataset', type=str, default='CIFAR10',
                        help='dataset, possible are `CIFAR10`, `CIFAR100`')
    parser.add_argument('--n_clients', type=int, default=10)
    parser.add_argument('--train_frac', help='fraction of train samples', type=float, default=0.8)
    parser.add_argument('--val_frac', help='fraction of validation samples in train samples', type=float, default=0.2)      
    parser.add_argument('--pathological_split',help='if selected, the dataset will be split as in' \
        '"Communication-Efficient Learning of Deep Networks from Decentralized Data";'
             'i.e., each client will receive `n_shards` of dataset, where each shard contains at most two classes',
        action='store_true'
    )
    parser.add_argument('--n_shards',help='number of shards given to each clients/task; ignored if \
        `--pathological_split` is not used; default is 2', type=int, default=2)
    parser.add_argument('--alpha', help = 'the parameter of dirichalet', type=float, default=1.0)

    # train part
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=4096, help='batch size') # 32
    parser.add_argument('--cuda', type=bool, default=tf.test.is_gpu_available())
    parser.add_argument('--gpu', type=str, default='0', help='use of gpu')
    parser.add_argument('--log_dir', type=str, default='log', help='directory of logs')
    parser.add_argument('--frac', type=float, default=1, help='Fraction of participating clients')
    parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
    parser.add_argument('--local_epoch', type=int, default=3, help='Number of local training epochs.')
    parser.add_argument('--eval_interval', type=int, default=1, help='Interval of evalution')
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    args = parser.parse_args()
    return args


def seed_everything(args):
    random.seed(args.seed)
    tf.random.set_random_seed(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    
    
def init_logger(args):
    log_file = os.path.join(args.log_dir, args.dataset + '.log')

    logging.basicConfig(
        format='%(asctime)s | %(message)s',
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        filename=log_file,
        filemode='w+'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

     
def main():
    args = arg_parse()

    seed_everything(args)

    init_logger(args)
      
    client_train_datasets, client_valid_datasets, client_test_datasets, data_info = load_dataset(args)

    clients = Clients(lambda graph: ConvNet(graph, input_size=data_info["num_features"], num_classes=data_info["num_classes"], \
            num_channels=data_info["num_channels"], learning_rate=args.lr, args=args), client_train_datasets, client_valid_datasets, client_test_datasets, data_info)
    
    run_fl(clients, args)    
 

if __name__ == "__main__":
    main()
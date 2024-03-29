# -*- coding: utf-8 -*-
from utils import logging
import torch.multiprocessing as mp
import torch
from client import Client
from server import Server
import torch.distributed.rpc as rpc
import os
from utils import load_dataset, init_clients_weights
from model import resnet20


def test_logging(test_logs, log_file):
    avg_test_val = 0.0
    for test_val in test_logs:
        avg_test_val += test_val

    logging("Final - all client - Test ACC: %.4f" % avg_test_val, log_file)


def run_train_eval(c_id, client, args):
    client.train(c_id, args)
    test_res = client.evaluation(c_id, args, mode="test")
    return test_res


def run_server(clients_name, args):
    client_train_datasets, client_valid_datasets, client_test_datasets, \
        data_info = load_dataset(args)

    server = Server(lambda: resnet20(
        in_channels=data_info["num_channels"],
        num_classes=data_info["num_classes"]), args)
    server_rref = rpc.RRef(server)

    clients = [Client(server_rref, lambda: resnet20(
        in_channels=data_info["num_channels"],
        num_classes=data_info["num_classes"]),
        lambda x: torch.optim.SGD(x, lr=args.lr, momentum=0.9), args,
        client_train_datasets[c_id], client_valid_datasets[c_id],
        client_test_datasets[c_id]) for c_id in range(args.n_clients)]

    # Initialize the aggretation weight
    init_clients_weights(clients)

    futs = []
    for c_id, client_name in enumerate(clients_name):
        futs.append(
            rpc.rpc_async(client_name, run_train_eval,
                          args=(c_id, clients[c_id], args))
        )

    torch.futures.wait_all(futs)

    with open(os.path.join(args.log_dir, args.dataset + ".log"), "a+") as log_file:
        test_logging([fut.value() for fut in futs], log_file)


def run(rank, world_size, args):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0  # Infinite timeout
    )
    if rank == 0:  # The process with rank 0 acts as the server
        rpc.init_rpc(
            "server",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # The default client ID starts from 0, so the client ID here is r - 1
        run_server([f"client{r - 1}" for r in range(1, world_size)], args)
    else:  # Other processes act as clients
        rpc.init_rpc(
            # The default client ID starts from 0, so the client ID here is r - 1
            f"client{rank - 1}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        # Trainer passively waiting for server to kick off training iterations

    # Block until all rpcs finish
    rpc.shutdown()


def run_fl(args):
    world_size = args.n_clients + 1
    mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)

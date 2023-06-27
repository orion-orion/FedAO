# -*- coding: utf-8 -*-
import time
import logging
import torch
import torch.multiprocessing as mp


def training_logging(train_logs, epoch, args):
    for c_id, train_val in enumerate(train_logs):
        logging.info("Global epoch {}/{} - client {} -  Training Loss: {:.3f}".format(
            epoch, args.global_epochs, c_id, train_val))


def evaluation_logging(eval_logs, epoch, mode="valid"):
    if mode == "valid":
        logging.info("Global epoch%d Valid:" % epoch)
    else:
        logging.info("Test:")

    avg_eval_val = 0.0
    for eval_val in eval_logs:
        avg_eval_val += eval_val.data

    logging.info("ACC: " + str(avg_eval_val))


def run_fl(clients, server, args):
    mp.set_start_method("spawn", force=True)

    begin = time.time()
    # Train with these clients
    for epoch in range(1, args.global_epochs + 1):
        # Choose some clients that will train on this epoch
        random_clients = server.choose_clients(args.n_clients, args.frac)

        # Train with these clients
        # Restore global weights to client's model
        global_weights = server.get_global_weights()
        processes = []
        train_logs = torch.tensor(
            [0.0 for _ in range(args.n_clients)]).share_memory_()
        for c_id in random_clients:
            # Restore global weights to client's model
            clients[c_id].set_global_weights(global_weights)

            p = mp.Process(target=clients[c_id].train_epoch, args=(
                c_id, train_logs, args, global_weights))
            # Train the model across `num_processes` processes
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        training_logging(list(train_logs), epoch, args)

        # Sum up active clients' weights at each epoch
        server.aggregate_weights([clients[c_id] for c_id in random_clients])

        # Evaluation on valid dataset
        if epoch % args.eval_interval == 0:
            eval_logs = torch.tensor(
                [0.0 for _ in range(args.n_clients)]).share_memory_()
            processes = []
            for c_id in range(args.n_clients):
                # Restore global weights to client's model
                clients[c_id].set_global_weights(server.get_global_weights())
                p = mp.Process(target=clients[c_id].evaluation, args=(
                    c_id, random_clients, eval_logs, "valid"))
                # Evaluate the model across `num_processes` processes
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            evaluation_logging(list(eval_logs), epoch, "valid")
    end = time.time()
    print("Time : %.4f" % (end - begin))

    # Evaluation on test dataset
    eval_logs = []
    processes = []
    eval_logs = torch.tensor(
        [0.0 for _ in range(args.n_clients)]).share_memory_()
    processes = []
    for c_id in range(args.n_clients):
        # Restore global weights to client's model
        clients[c_id].set_global_weights(server.get_global_weights())
        p = mp.Process(target=clients[c_id].evaluation, args=(
            c_id, random_clients, eval_logs, "test"))
        # Evaluate the model across `num_processes` processes
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    evaluation_logging(list(eval_logs), epoch, "test")

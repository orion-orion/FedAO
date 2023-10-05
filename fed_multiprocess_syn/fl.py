# -*- coding: utf-8 -*-
import time
import logging
import torch
import torch.multiprocessing as mp


def training_logging(train_logs, round, args):
    for c_id, train_val in enumerate(train_logs):
        logging.info("Training round {}/{} - client {} -  Training Loss:"
                     "{:.3f}".format(round, args.rounds, c_id, train_val))


def evaluation_logging(eval_logs, round, mode="valid"):
    if mode == "valid":
        logging.info("Training round%d Valid:" % round)
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
    for round in range(1, args.rounds + 1):
        # Choose some clients that will train in this round
        random_clients = server.choose_clients(args.n_clients, args.frac)

        # Train with these clients
        # Restore global parameters to client's model
        global_params = server.get_global_params()
        processes = []
        train_logs = torch.tensor(
            [0.0 for _ in range(args.n_clients)]).share_memory_()
        for c_id in random_clients:
            # Restore global parameters to client's model
            clients[c_id].set_params(global_params)

            p = mp.Process(target=clients[c_id].train_epochs, args=(
                c_id, train_logs, args, global_params))
            # Train the model across `num_processes` processes
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
        training_logging(list(train_logs), round, args)

        # Sum up active clients' parameters in each round
        server.aggregate_params([clients[c_id] for c_id in random_clients])

        # Evaluation on valid dataset
        if round % args.eval_interval == 0:
            eval_logs = torch.tensor(
                [0.0 for _ in range(args.n_clients)]).share_memory_()
            processes = []
            for c_id in range(args.n_clients):
                if not args.fed_method == "Ditto":
                    # Restore global parameters to client's model
                    clients[c_id].set_params(server.get_global_params())
                p = mp.Process(target=clients[c_id].evaluation, args=(
                    c_id, random_clients, eval_logs, args, "valid"))
                # Evaluate the model across `num_processes` processes
                p.start()
                processes.append(p)

            for p in processes:
                p.join()
            evaluation_logging(list(eval_logs), round, "valid")
    end = time.time()
    print("Time : %.4f" % (end - begin))

    # Evaluation on test dataset
    eval_logs = []
    processes = []
    eval_logs = torch.tensor(
        [0.0 for _ in range(args.n_clients)]).share_memory_()
    processes = []
    for c_id in range(args.n_clients):
        if not args.fed_method == "Ditto":
            # Restore global parameters to client's model
            clients[c_id].set_params(server.get_global_params())
        p = mp.Process(target=clients[c_id].evaluation, args=(
            c_id, random_clients, eval_logs, args, "test"))
        # Evaluate the model across `num_processes` processes
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    evaluation_logging(list(eval_logs), round, "test")

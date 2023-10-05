# -*- coding: utf-8 -*-
import logging
from tqdm import tqdm


def evaluation_logging(eval_logs, round, mode="valid"):
    if mode == "valid":
        logging.info("Training round%d Valid:" % round)
    else:
        logging.info("Test:")

    avg_eval_log = {}
    for key in eval_logs[0].keys():
        avg_eval_val = 0
        for i in range(len(eval_logs)):
            avg_eval_val += eval_logs[i][key]
        avg_eval_log[key] = avg_eval_val
    logging.info("ACC: " + str(avg_eval_log["ACC"]))


def run_fl(clients, args):
    global_params = clients.get_params()
    if args.fed_method == "Ditto":
        # Personalized model parameters for Ditto
        client_per_params = [clients.get_params() for i in range(args.n_clients)]    
    for round in range(1, args.rounds + 1):
        # We are going to sum up active clients" parameters in each round
        client_params_sum = None

        # Choose some clients that will train in this round
        random_clients = clients.choose_clients(args.frac)

        # Train with these clients
        for c_id in tqdm(random_clients, ascii=True):
            # Restore global parameters to client's model
            clients.set_params(global_params)

            # Train one client
            clients.train_epochs(c_id, round, args, global_params)

            # Obtain current client's parameters
            current_client_params = clients.get_params()

            # Sum it up with parameters
            if client_params_sum is None:
                client_params_sum = [clients.client_train_weight[c_id] * x
                                      for x in current_client_params]
            else:
                for w_sum, w in zip(client_params_sum,
                                    current_client_params):
                    w_sum += clients.client_train_weight[c_id] * w

            if args.fed_method == "Ditto":
                # Updates personalized model parameters for Ditto
                clients.set_params(client_per_params[c_id])

                # Train the personalized model
                clients.train_epochs(c_id, round, args, global_params, per=True)

                # Obtain and store current client's parameters
                client_per_params[c_id] = clients.get_params()                    

        # Assign the avg parameters to global parameters
        global_params = client_params_sum
        if args.valid_frac > 0 and round % args.eval_interval == 0:
            eval_logs = []
            for c_id in tqdm(range(args.n_clients), ascii=True):
                if args.fed_method == "Ditto":
                    clients.set_params(client_per_params[c_id])
                elif c_id == 0:
                    clients.set_params(global_params)                
                if c_id in random_clients:
                    eval_log = clients.evaluation(c_id, args, mode="valid")
                else:
                    eval_log = clients.get_old_eval_log()
                eval_logs.append(dict((key, value
                                       * clients.client_valid_weight[c_id])
                                      for key, value in eval_log.items()))

            evaluation_logging(eval_logs, round, mode="valid")

    eval_logs = []
    for c_id in tqdm(range(args.n_clients), ascii=True):
        if args.fed_method == "Ditto":
            clients.set_params(client_per_params[c_id])
        elif c_id == 0:
            clients.set_params(global_params)        
        eval_log = clients.evaluation(c_id, args, mode="test")
        eval_logs.append(dict((key, value * clients.client_test_weight[c_id])
                              for key, value in eval_log.items()))

    evaluation_logging(eval_logs, round, mode="test")

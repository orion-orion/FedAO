import logging
from tqdm import tqdm

def evaluation_logging(eval_logs, epoch, mod="valid"):
    if mod == "valid":
        logging.info('Global epoch%d Valid:' % epoch)
    else:
        logging.info('Test:')
        
    avg_eval_log = {}
    for key in eval_logs[0].keys():
        avg_eval_val = 0
        for i in range(len(eval_logs)):
            avg_eval_val += eval_logs[i][key]
        avg_eval_log[key] = avg_eval_val
        
    logging.info('ACC: ' + str(avg_eval_log["ACC"]))
       
       
def run_fl(clients, args):
    global_weights = clients.get_client_weights()
    # for c_id, train_data
    for epoch in range(1, args.global_epochs + 1):

        # We are going to sum up active clients' weights at each epoch
        client_weights_sum = None
         
        # participating_cids = server.select_clients(n_clients, opt['frac']) 
        random_clients = clients.choose_clients(args.frac)
            
        # Train with these clients
        for c_id in tqdm(random_clients, ascii=True):
            # Restore global weights to client's model
            clients.set_global_weights(global_weights)

            # train one client
            clients.train_epoch(c_id, epoch, args)

            # obtain current client's weights
            current_client_weights = clients.get_client_weights()

            # sum it up with weights
            if client_weights_sum is None:
                client_weights_sum = dict((key, value * clients.client_train_prop[c_id]) for key, value in current_client_weights.items())
            else:
                for key in client_weights_sum.keys():
                    client_weights_sum[key] += clients.client_train_prop[c_id] * current_client_weights[key]

        # assign the avg weights to global weights
        global_weights = client_weights_sum
        if args.val_frac > 0 and epoch % args.eval_interval == 0:
            clients.set_global_weights(global_weights)
            eval_logs = []
            for c_id in tqdm(range(args.n_clients), ascii=True):
                if c_id in random_clients:
                    eval_log = clients.evaluation(c_id, mod = "valid")
                else:
                    eval_log = clients.get_old_eval_log()
                eval_logs.append(dict((key, value * clients.client_valid_prop[c_id]) for key, value in eval_log.items()))

            evaluation_logging(eval_logs, epoch, mod="valid")
   
    clients.set_global_weights(global_weights)
    eval_logs = []
    for c_id in tqdm(range(args.n_clients), ascii=True):
        eval_log = clients.evaluation(c_id, mod = "test")
        eval_logs.append(dict((key, value * clients.client_test_prop[c_id]) for key, value in eval_log.items()))

    evaluation_logging(eval_logs, epoch, mod="test")
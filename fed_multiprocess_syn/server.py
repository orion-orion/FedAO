# -*- coding: utf-8 -*-
import math
import numpy as np


class Server(object):
    def __init__(self, model_fn, args):
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.model = model_fn().to(self.device)
        # Here `global_weights` is a reference, the parameters of the model change
        # with `self.global_weights`
        self.global_weights = {key: value for key,
                               value in self.model.named_parameters()}

    def aggregate_weights(self, clients):
        """Sums up active clients" weights at each epoch.
        """
        for c_id, client in enumerate(clients):
            current_client_weights = client.get_weights()
            if c_id == 0:
                for name in self.global_weights.keys():
                    self.global_weights[name].data = client.train_prop * \
                        current_client_weights[name].data.clone()
            else:
                for name in self.global_weights.keys():
                    self.global_weights[name].data += client.train_prop * \
                        current_client_weights[name].data.clone()

    def choose_clients(self, n_clients, ratio=1.0):
        """Randomly chooses some clients.
        """
        choose_num = math.ceil(n_clients * ratio)
        return np.random.permutation(n_clients)[:choose_num]

    def get_global_weights(self):
        """Returns a reference to the parameters of the global model.
        """
        return self.global_weights

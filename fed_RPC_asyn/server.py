# -*- coding: utf-8 -*-
import torch
import threading
import torch.distributed.rpc as rpc
from torch import optim


class Server(object):

    def __init__(self, model_fn, args):
        self.device = "cuda" if args.cuda else "cpu"
        self.model = model_fn()
        self.global_params = self.model.state_dict()
        self.n_clients = args.n_clients
        # The parameter is about merging the new model to the global model
        self.async_lam = args.async_lam

        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()

    def get_model(self):
        # TensorPipe RPC backend only supports CPU tensors,
        # so make sure your tensors are in CPU before sending them over RPC
        return self.global_params

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(server_rref, client_params):
        self = server_rref.local_value()
        for name, g_w in self.global_params.items():
            # Here is the simplest way to update the global model:
            # w^{t+1} = (1 - async_lam) * w^t + async_lam * w^{new}
            # it can be modified as needed
            self.global_params[name] = (
                1 - self.async_lam) * g_w + self.async_lam * client_params[name]
        self.model.load_state_dict(self.global_params)

        with self.lock:
            fut = self.future_model
            fut.set_result(self.global_params)
            self.future_model = torch.futures.Future()

        return fut

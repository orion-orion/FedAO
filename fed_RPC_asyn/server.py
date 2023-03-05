import torch
import threading
import torch.distributed.rpc as rpc

    
class Server(object):

    def __init__(self, model_fn, args):
        self.device = "cuda" if args.cuda else "cpu"
        self.model = model_fn()
        self.global_weights = self.model.state_dict()
        self.n_clients = args.n_clients
        self.lam = args.lam # The parameter is about merging the new model to the global model

        self.lock = threading.Lock()
        self.future_model = torch.futures.Future()

    def get_model(self):
        # TensorPipe RPC backend only supports CPU tensors, 
        # so make sure your tensors are in CPU before sending them over RPC
        return self.global_weights

    @staticmethod
    @rpc.functions.async_execution
    def update_and_fetch_model(server_rref, client_weights):
        self = server_rref.local_value()
        for name, g_w in self.global_weights.items():
            # 这里采用最简单的方式更新全局模型: w^{t+1} = (1 - lam) * w^t + lam * w^{new} 
            # 大家可以根据自己的需要修改
            self.global_weights[name] = (1 - self.lam) * g_w + self.lam * client_weights[name]
        self.model.load_state_dict(self.global_weights)

        with self.lock:
            fut = self.future_model
            fut.set_result(self.global_weights)
            self.future_model = torch.futures.Future()

        return fut



import numpy as np
import math


class Server(object):
    def __init__(self, model_fn, args):
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.model = model_fn().to(self.device)
        # 此处为传引用，model参数变化self.weights而变化，get_weights()函数也实际返回的weights的引用
        self.global_weights = {key : value for key, value in self.model.named_parameters()}

    def aggregate_weights(self, clients):
        # We are going to sum up active clients' weights at each epoch
        # print(clients)
        for c_id, client in enumerate(clients):
            current_client_weights = client.get_weights()
            if c_id == 0:
                # print("*******0*******", client.train_prop)
                for name in self.global_weights.keys():
                    self.global_weights[name].data = client.train_prop * current_client_weights[name].data.clone()
                # if name == "linear.bias":
                #     print(current_client_weights[name])
            else:
                # print("*******other*****", client.train_prop)
                for name in self.global_weights.keys():
                    self.global_weights[name].data += client.train_prop * current_client_weights[name].data.clone()
                    # if name == "linear.bias":
                    #     print(current_client_weights[name])

    def choose_clients(self, n_clients, ratio=1.0):
        """ randomly choose some clients """
        choose_num = math.ceil(n_clients * ratio)
        return np.random.permutation(n_clients)[:choose_num]
    
    def get_global_weights(self):
        return self.global_weights


import numpy as np
import math
import gc
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader
class Clients:
    def __init__(self, model_fn, optimizer_fn, args, train_datasets, test_datasets, valid_datasets):
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.model = model_fn().to(self.device)
        self.optimizer = optimizer_fn(self.model.parameters())
        
        self.train_dataloaders = [DataLoader(train_datasets[c_id], batch_size=args.batch_size, shuffle=True) for c_id in range(args.n_clients)]
        self.valid_dataloaders = [DataLoader(valid_datasets[c_id], batch_size=args.batch_size, shuffle=False) for c_id in range(args.n_clients)]
        self.test_dataloaders = [DataLoader(test_datasets[c_id], batch_size=args.batch_size, shuffle=False) for c_id in range(args.n_clients)]

        # 初始化各client的权重
        client_n_samples_train = [len(train_dataloader) for train_dataloader in self.train_dataloaders]
        samples_sum_train = sum(client_n_samples_train)
        self.client_train_prop = [len(train_dataloader)/samples_sum_train for train_dataloader in self.train_dataloaders]

        client_n_samples_valid = [len(valid_dataloader) for valid_dataloader in self.valid_dataloaders]
        samples_sum_valid = sum(client_n_samples_valid)
        self.client_valid_prop = [len(valid_dataloader)/samples_sum_valid for valid_dataloader in self.valid_dataloaders]
        
        client_n_samples_test = [len(test_dataloader) for test_dataloader in self.test_dataloaders]
        samples_sum_test = sum(client_n_samples_test)
        self.client_test_prop = [len(test_dataloader)/samples_sum_test for test_dataloader in self.test_dataloaders]

    def train_epoch(self, c_id, epoch, args):
        """
            Train one client with its own data for one epoch
            cid: Client id
        """
        for _ in range(args.local_epochs):
            loss, step, n_samples = 0.0, 0, 0
            for x, y in self.train_dataloaders[c_id]:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                # Pytorch自带softmax和默认取了mean
                l = torch.nn.CrossEntropyLoss()(self.model(x), y)

                l.backward()
                self.optimizer.step()  
                
                loss += l.item() * y.shape[0]
                n_samples += y.shape[0]
            gc.collect()
        logging.info('Global epoch {}/{} - client {} -  Training Loss: {:.3f}'.format(epoch, args.global_epochs, c_id, loss / n_samples))
        return n_samples
    
    def evaluation(self, c_id, mod = "valid"):    
        if mod == "valid":
            dataloader = self.valid_dataloaders[c_id]
        elif mod == "test":
            dataloader = self.test_dataloaders[c_id]

        n_samples, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)
                
                y_hat = self.model(x)
                pred = torch.argmax(y_hat.data, 1)

                correct += (pred == y).sum().item()
                n_samples += y.shape[0]

        gc.collect()
        self.acc = correct/n_samples
        return {"ACC": self.acc}
         
    def get_old_eval_log(self):
        return {"ACC": self.acc}

    def get_client_weights(self):
        """ Return all of the weights list """
        return self.model.state_dict()

    def set_global_weights(self, global_weights):
        """ Assign all of the weights with global weights """
        self.model.load_state_dict(global_weights)

    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients """
        client_num = self.get_clients_num()
        choose_num = math.ceil(client_num * ratio)
        return np.random.permutation(client_num)[:choose_num]

    def get_clients_num(self):
        return len(self.train_dataloaders)

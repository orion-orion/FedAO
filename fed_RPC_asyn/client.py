import gc
import torch
from torch.utils.data import DataLoader
import os
# import logging
from server import Server
import torch.distributed.rpc as rpc
from utils import logging

    
class Client(object):

    def __init__(self, ps_rref, model_fn, optimizer_fn, args, train_dataset, test_dataset, valid_dataset):
        self.device = "cuda" if args.cuda else "cpu"
        self.model = model_fn() # 初始化由于要参与RPC通信，暂时先不分配在GPU，后面训练的时候再分配
        self.optimizer = optimizer_fn(self.model.parameters())
        self.ps_rref = ps_rref
            
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)    
        
        # 初始化各client的样本占所有样本的比率
        self.train_pop, self.valid_prop, self.test_prop = 0.0, 0.0, 0.0
        
        # 初始化各client的权重
        self.n_samples_train = len(train_dataset)
        if args.valid_frac > 0:
            self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)
            
    def train(self, c_id, args):
        """
            Train one client with its own data for one epoch
            cid: Client id
            args: Arguments of training
        """
        self.log_file = open(os.path.join(args.log_dir, args.dataset + '.log'), "a+")

        torch.manual_seed(args.seed + c_id)
        self.model.train()
        for epoch in range(1, args.global_epochs + 1):
            global_weights = rpc.rpc_sync(
                self.ps_rref.owner(),
                Server.update_and_fetch_model,
                args=(self.ps_rref, self.get_client_weights()), # 注意：RPC只支持CPU通信
            )
            self.set_global_weights(global_weights)
            self.model.to(self.device) # 此时将模型分配在GPU
                 
            for _ in range(args.local_epochs):
                loss, n_samples = 0.0, 0
                for x, y in self.train_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()

                    # Pytorch自带softmax和默认取了mean
                    l = torch.nn.CrossEntropyLoss()(self.model(x), y)
                    l.backward()
                    self.optimizer.step()

                    loss += l.item() * y.shape[0]
                    n_samples += y.shape[0]
                
                gc.collect()
            logging('Global epoch {}/{} - client {} -  Training Loss: {:.3f}'.format(epoch, args.global_epochs, c_id, loss/n_samples), \
                self.log_file)   
                
            if args.valid_frac > 0 and epoch % args.eval_interval == 0:
                self.evaluation(c_id, args, epoch, mod="valid")
    
    def evaluation(self, c_id, args, epoch=0, mod="valid"):   
        if mod == "valid":
            dataloader = self.valid_dataloader
        elif mod == "test":
            dataloader = self.test_dataloader

        self.model.eval()
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

        if mod == "valid":
            logging('Global epoch {}/{} - client {} -  Valid ACC: {:.4f}'.format(epoch, args.global_epochs, c_id, self.acc), self.log_file)   
        else:
            logging('Final - client {} -  Test ACC: {:.4f}'.format(c_id, self.acc), self.log_file) 
        
        if mod == "valid":
            return self.acc * self.valid_prop
        elif mod == "test":
            return self.acc * self.test_prop

    def get_client_weights(self):
        """ Return all of the weights list """
        return self.model.cpu().state_dict()
    
    def set_global_weights(self, global_weights):
        """ Assign all of the weights with global weights """
        self.model.load_state_dict(global_weights)







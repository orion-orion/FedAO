import gc
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

class Client:
    def __init__(self, model_fn, optimizer_fn, args, train_dataset, test_dataset, valid_dataset):
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.model = model_fn().to(self.device)
        self.optimizer = optimizer_fn(self.model.parameters())
        
        self.train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, \
                collate_fn=self.collate_fn)
        self.valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, \
                collate_fn=self.collate_fn)
        self.test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, \
                collate_fn=self.collate_fn)

        # 此处为传引用，model参数变化self.weights而变化，get_weights()函数也实际返回的weights的引用
        self.weights = {key : value for key, value in self.model.named_parameters()}
        
        # 初始化各client的样本占所有样本的比率
        self.train_pop, self.valid_prop, self.test_prop = 0.0, 0.0, 0.0
        
        # 初始化各client的权重
        self.n_samples_train = len(train_dataset)
        if args.valid_frac > 0:
            self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)
        
    def train_epoch(self, c_id, train_logs, args):
        """
            Train one client with its own data for one epoch
            cid: Client id
        """
        torch.manual_seed(args.seed + c_id)
        self.model.train()
        for _ in range(args.local_epochs):
            loss, n_samples = 0.0, 0
            for x, y in self.train_dataloader:
                # x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                # Pytorch自带softmax和默认取了mean
                l = torch.nn.CrossEntropyLoss()(self.model(x), y)
                l.backward()
                self.optimizer.step()  
                
                loss += l.item() * y.shape[0]
                n_samples += y.shape[0]
                
            gc.collect()
        train_logs[c_id] = float(loss / n_samples)

    def evaluation(self, c_id, random_clients, eval_logs, mod="valid"):   
        if c_id not in random_clients and mod == "valid":
                eval_logs[c_id] = self.get_old_eval_log()
        else:
            if mod == "valid":
                dataloader = self.valid_dataloader
            elif mod == "test":
                dataloader = self.test_dataloader

            self.model.eval()
            n_samples, correct = 0, 0
            with torch.no_grad():
                for x, y in dataloader:
                    # x, y = x.to(self.device), y.to(self.device)
                    
                    y_hat = self.model(x)
                    pred = torch.argmax(y_hat.data, 1)

                    correct += (pred == y).sum().item()
                    n_samples += y.shape[0]

            gc.collect()
            self.acc = correct/n_samples
            if mod == "valid":
                eval_logs[c_id] = float(self.acc * self.valid_prop)
            elif mod == "test":
                eval_logs[c_id] = float(self.acc * self.test_prop)
                     
    def get_old_eval_log(self):
        return self.acc

    def get_weights(self):
        """ Return all of the weights list """
        return self.weights

    def set_global_weights(self, global_weights):
        """ Assign all of the weights with global weights """
        for name in self.weights.keys():
            self.weights[name].data = global_weights[name].data.clone()

    def collate_fn(self, batch):
        return tuple(x.to(self.device) for x in default_collate(batch))
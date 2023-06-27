# -*- coding: utf-8 -*-
import gc
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate


class Client:
    def __init__(self, model_fn, optimizer_fn, args, train_dataset,
                 test_dataset, valid_dataset):
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.model = model_fn().to(self.device)
        self.optimizer = optimizer_fn(self.model.parameters())

        # Here `collate_fn` is used to move data to `self.device`
        self.train_dataloader = DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           collate_fn=self.collate_fn)
        # Here `collate_fn` is used to move data to `self.device`
        self.valid_dataloader = DataLoader(valid_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           collate_fn=self.collate_fn)
        # Here `collate_fn` is used to move data to `self.device`
        self.test_dataloader = DataLoader(test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          collate_fn=self.collate_fn)

        # Here `self.weights` is a reference of the model parameters, and the
        # parameters of the model change with `self.weights`
        self.weights = {key: value for key,
                        value in self.model.named_parameters()}

        # Initialize the proportion of the samples of each client to
        # all samples
        self.train_pop, self.valid_prop, self.test_prop = 0.0, 0.0, 0.0

        # Compute the number of samples for each client
        self.n_samples_train = len(train_dataset)
        if args.valid_frac > 0:
            self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)

    def train_epoch(self, c_id, train_logs, args, global_weights):
        """Train one client with its own data for one epoch.

        Args:
            c_id: client ID.
            train_logs: training logs.
            args: other parameters for training.
            global_weights: global model weights used in `FedProx` method.
        """
        torch.manual_seed(args.seed + c_id)
        self.model.train()
        for _ in range(args.local_epochs):
            loss, n_samples = 0.0, 0
            for x, y in self.train_dataloader:
                # x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # The `CrossEntropyLoss` function in Pytorch will divide by the
                # mini-batch size by default
                l = torch.nn.CrossEntropyLoss()(self.model(x), y)
                if args.fed_method == "FedProx":
                    l += self.prox_reg(dict(self.model.named_parameters()),
                                       global_weights, args.mu)

                l.backward()
                self.optimizer.step()

                loss += l.item() * y.shape[0]
                n_samples += y.shape[0]

            gc.collect()
        train_logs[c_id] = float(loss / n_samples)

    @ staticmethod
    def flatten(params):
        return torch.cat([param.flatten() for param in params])

    def prox_reg(self, params1, params2, mu):
        # The parameters returned by `named_parameters()` should be rearranged
        # according to the keys of the parameters returned by `state_dict()`
        params2_values = [params2[key] for key in params1.keys()]

        # Multi-dimensional parameters should be flattened into one-dimensional
        vec1 = self.flatten(params1.values())
        vec2 = self.flatten(params2_values)
        return mu / 2 * torch.norm(vec1 - vec2)**2

    def evaluation(self, c_id, random_clients, eval_logs, mode="valid"):
        """Evaluation one client with its own data for one epoch.

        Args:
            c_id: client ID.
            random_clients: randomly selected clients.
            eval_logs: evaluation logs.
            mode: choose valid or test mode.
        """
        if c_id not in random_clients and mode == "valid":
            eval_logs[c_id] = self.get_old_eval_log()
        else:
            if mode == "valid":
                dataloader = self.valid_dataloader
            elif mode == "test":
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
            if mode == "valid":
                eval_logs[c_id] = float(self.acc * self.valid_prop)
            elif mode == "test":
                eval_logs[c_id] = float(self.acc * self.test_prop)

    def get_old_eval_log(self):
        """Returns the evaluation result of the lastest epoch.
        """
        return self.acc

    def get_weights(self):
        """Returns all of the weights in `dict` format. Note that
        `self.weights` is a reference of the model parameters.
        """
        return self.weights

    def set_global_weights(self, global_weights):
        """Assigns all of the weights with global weights.
        """
        for name in self.weights.keys():
            self.weights[name].data = global_weights[name].data.clone()

    def collate_fn(self, batch):
        """Moves data to `self.device`.
        """
        return tuple(x.to(self.device) for x in default_collate(batch))

# -*- coding: utf-8 -*-
import math
import gc
import copy
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader


class Clients:
    def __init__(self, model_fn, optimizer_fn, args, train_datasets,
                 valid_datasets, test_datasets):
        self.device = "cuda:%s" % args.gpu if args.cuda else "cpu"
        self.model = model_fn().to(self.device)
        self.optimizer = optimizer_fn(self.model.parameters())
        self.n_clients = args.n_clients

        self.train_dataloaders = [DataLoader(train_datasets[c_id],
                                             batch_size=args.batch_size,
                                             shuffle=True)
                                  for c_id in range(args.n_clients)]
        self.valid_dataloaders = [DataLoader(valid_datasets[c_id],
                                             batch_size=args.batch_size,
                                             shuffle=False)
                                  for c_id in range(args.n_clients)]
        self.test_dataloaders = [DataLoader(test_datasets[c_id],
                                            batch_size=args.batch_size,
                                            shuffle=False)
                                 for c_id in range(args.n_clients)]

        # Initialize the aggretation weight
        client_n_samples_train = [len(train_datasets[c_id])
                                  for c_id in range(args.n_clients)]
        samples_sum_train = sum(client_n_samples_train)
        self.client_train_weight = [
            len(train_datasets[c_id])/samples_sum_train
            for c_id in range(args.n_clients)]
        if args.valid_frac > 0:
            client_n_samples_valid = [len(valid_datasets[c_id])
                                      for c_id in range(args.n_clients)]
            samples_sum_valid = sum(client_n_samples_valid)
            self.client_valid_weight = [len(valid_datasets[c_id])
                                        / samples_sum_valid
                                        for c_id in range(args.n_clients)]
        client_n_samples_test = [len(test_datasets[c_id])
                                 for c_id in range(args.n_clients)]
        samples_sum_test = sum(client_n_samples_test)
        self.client_test_weight = [len(test_datasets[c_id]) / samples_sum_test
                                   for c_id in range(args.n_clients)]

    def train_epochs(self, c_id, round, args, global_params, per=False):
        """Train one client with its own data for local epochs.

        Args:
            c_id: Client ID.
            round: Current training round.
            args: Other parameters for training.
            global_params: Global model parameters used in `FedProx` of `Ditto`
                method.
            per: A flag varaible indicating whether a personalized model in
                `Ditto` method is being trained.
        """
        self.model.train()
        for _ in range(args.local_epochs):
            loss, n_samples = 0.0, 0
            for x, y in self.train_dataloaders[c_id]:
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                # The `CrossEntropyLoss` function in Pytorch will divide by
                # the mini-batch size by default
                batch_loss = torch.nn.CrossEntropyLoss()(self.model(x), y)
                if args.fed_method == "FedProx":
                    batch_loss += self.prox_reg(dict(
                        self.model.named_parameters()),
                        global_params, args.mu)
                elif args.fed_method == "Ditto" and per:
                    batch_loss += self.prox_reg(dict(
                        self.model.named_parameters()),
                        global_params, args.lam)

                batch_loss.backward()
                self.optimizer.step()

                loss += batch_loss.item() * y.shape[0]
                n_samples += y.shape[0]

            gc.collect()
        if not (args.fed_method == "Ditto") or per:
            logging.info("Training round {}/{} - client {} -  Training Loss: "
                         "{:.3f}".format(round, args.rounds, c_id,
                                         loss / n_samples))
        return n_samples

    @ staticmethod
    def flatten(params):
        return torch.cat([param.flatten() for param in params])

    def prox_reg(self, params1, params2, weight_factor):
        # The parameters returned by `named_parameters()` should be rearranged
        # according to the keys of the parameters returned by `state_dict()`
        params2_values = [params2[key] for key in params1.keys()]

        # Multi-dimensional parameters should be flattened into one-dimensional
        vec1 = self.flatten(params1.values())
        vec2 = self.flatten(params2_values)
        return weight_factor / 2 * torch.norm(vec1 - vec2)**2

    def evaluation(self, c_id, mode="valid"):
        """Evaluation one client with its own data.

        Args:
            c_id: Client ID.
            mode: Choose valid or test mode.
        """
        if mode == "valid":
            dataloader = self.valid_dataloaders[c_id]
        elif mode == "test":
            dataloader = self.test_dataloaders[c_id]

        self.model.eval()
        n_samples, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                logits = self.model(x)
                pred = torch.argmax(logits.data, 1)

                correct += (pred == y).sum().item()
                n_samples += y.shape[0]

        gc.collect()
        self.acc = correct/n_samples
        return {"ACC": self.acc}

    def get_old_eval_log(self):
        """Returns the evaluation result of the lastest round.
        """
        return {"ACC": self.acc}

    def get_params(self):
        """Returns all of the parameters in `OrderedDict` format. Note that
        the `state_dict()` function returns the reference of the model
        parameters, so here we use deep copy.
        """
        return copy.deepcopy(self.model.state_dict())

    def set_params(self, params):
        """Assigns all of the old parameters with new parameters.
        """
        self.model.load_state_dict(params)

    def choose_clients(self, ratio=1.0):
        """Randomly chooses some clients.
        """
        choose_num = math.ceil(self.n_clients * ratio)
        return np.random.permutation(self.n_clients)[:choose_num]

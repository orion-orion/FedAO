# -*- coding: utf-8 -*-
import gc
import torch
from torch.utils.data import DataLoader
import os
from server import Server
import torch.distributed.rpc as rpc
from utils import logging


class Client(object):

    def __init__(self, server_rref, model_fn, optimizer_fn, args,
                 train_dataset, test_dataset, valid_dataset):
        self.device = "cuda" if args.cuda else "cpu"
        # Since the model parameters need to participate in RPC communication,
        # the model is not allocated on the GPU when it is initialized, and
        # then moved to the GPU when it is trained later
        self.model = model_fn()
        self.per_model = model_fn()
        self.optimizer = optimizer_fn(self.model.parameters())
        self.per_optimizer = optimizer_fn(self.per_model.parameters())
        self.server_rref = server_rref

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True)
        self.valid_dataloader = DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=False)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False)

        # The aggretation weight
        self.train_pop, self.valid_weight, self.test_weight = 0.0, 0.0, 0.0

        # Compute the number of samples for each client
        self.n_samples_train = len(train_dataset)
        if args.valid_frac > 0:
            self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)

    def train(self, c_id, args):
        """Train one client with its own data for local epochs.

        Args:
            c_id: Client ID.
            args: Other parameters for training.
        """
        self.log_file = open(os.path.join(
            args.log_dir, args.dataset + ".log"), "a+")

        torch.manual_seed(args.seed + c_id)
        self.model.train()
        for round in range(1, args.rounds + 1):
            global_params = rpc.rpc_sync(
                self.server_rref.owner(),
                Server.update_and_fetch_model,
                # Note that RPC only supports CPU
                args=(self.server_rref, self.get_params()),
            )
            self.set_params(global_params)
            self.model.to(self.device)  # move the model to the GPU
            self.per_model.to(self.device)  # move the model to the GPU

            for _ in range(args.local_epochs):
                loss, n_samples = 0.0, 0
                for x, y in self.train_dataloader:
                    x, y = x.to(self.device), y.to(self.device)
                    self.optimizer.zero_grad()

                    # The `CrossEntropyLoss` function in Pytorch will divide
                    # by the mini-batch size by default
                    batch_loss = torch.nn.CrossEntropyLoss()(self.model(x), y)
                    if args.fed_method == "FedProx":
                        batch_loss += self.prox_reg(dict(
                            self.model.named_parameters()),
                            global_params, args.mu)

                    batch_loss.backward()
                    self.optimizer.step()

                    if args.fed_method == "Ditto":
                        # Update the personalized model
                        self.per_optimizer.zero_grad()

                        batch_per_loss = torch.nn.CrossEntropyLoss()(
                            self.per_model(x), y)
                        batch_per_loss += self.prox_reg(dict(
                            self.per_model.named_parameters()),
                            global_params, args.lam)

                        batch_per_loss.backward()
                        self.per_optimizer.step()

                        loss += batch_per_loss.item() * y.shape[0]
                    else:
                        loss += batch_loss.item() * y.shape[0]

                    n_samples += y.shape[0]

                gc.collect()
            logging("Training round {}/{} - client {} -  Training Loss: {:.3f}"
                    .format(round, args.rounds, c_id, loss/n_samples),
                    self.log_file)

            if args.valid_frac > 0 and round % args.eval_interval == 0:
                self.evaluation(c_id, args, round, mode="valid")

    @ staticmethod
    def flatten(params):
        return torch.cat([param.flatten() for param in params])

    def prox_reg(self, params1, params2, weight_factor):
        # The parameters returned by `named_parameters()` should be rearranged
        # according to the keys of the parameters returned by `state_dict()`.
        # Note that params2 should be moved to the device used first
        params2_values = [params2[key].to(self.device)
                          for key in params1.keys()]

        # Multi-dimensional parameters should be flattened into one-dimensional
        vec1 = self.flatten(params1.values())
        vec2 = self.flatten(params2_values)
        return weight_factor / 2 * torch.norm(vec1 - vec2)**2

    def evaluation(self, c_id, args, epoch=0, mode="valid"):
        """Evaluation one client with its own data.

        Args:
            c_id: Client ID.
            args: Evaluation arguments.
            round: Current training round.
            mode: Choose valid or test mode.
        """
        if mode == "valid":
            dataloader = self.valid_dataloader
        elif mode == "test":
            dataloader = self.test_dataloader

        self.model.eval()
        n_samples, correct = 0, 0
        with torch.no_grad():
            for x, y in dataloader:
                x, y = x.to(self.device), y.to(self.device)

                if args.fed_method == "Ditto":
                    y_hat = self.model(x)
                else:
                    y_hat = self.model(x)

                pred = torch.argmax(y_hat.data, 1)

                correct += (pred == y).sum().item()
                n_samples += y.shape[0]

        gc.collect()
        self.acc = correct/n_samples

        if mode == "valid":
            logging("Training round {}/{} - client {} -  Valid ACC: {:.4f}".format(epoch,
                    args.rounds, c_id, self.acc), self.log_file)
        else:
            logging("Final - client {} -  Test ACC: {:.4f}".format(c_id,
                    self.acc), self.log_file)

        if mode == "valid":
            return self.acc * self.valid_weight
        elif mode == "test":
            return self.acc * self.test_weight

    def get_params(self):
        """Returns all of the parameters in `OrderedDict` format. Note that
        the `state_dict()` function returns the reference of the model
        parameters. In consideration of computational efficiency, here we
        choose to keep the reference.In addition, it should be noted that
        RPC only supports CPU, so we need to move the model to CPU first.
        """
        return self.model.cpu().state_dict()

    def set_params(self, global_params):
        """Assigns all of the parameters with global parameters.
        """
        self.model.load_state_dict(global_params)

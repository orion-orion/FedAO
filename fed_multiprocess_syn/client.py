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
        self.per_model = model_fn().to(self.device)
        self.optimizer = optimizer_fn(self.model.parameters())
        self.per_optimizer = optimizer_fn(self.per_model.parameters())

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

        # Here `self.params` is a reference of the model parameters, and the
        # parameters of the model change with `self.params`
        self.params = {key: value for key,
                       value in self.model.named_parameters()}
        # Here `self.params` is a reference of the personalized model
        # parameters, and the parameters of the personalized model change with
        # `self.per_params`
        self.per_params = {key: value for key,
                           value in self.per_model.named_parameters()}

        # The aggretation weight
        self.train_pop, self.valid_weight, self.test_weight = 0.0, 0.0, 0.0

        # Compute the number of samples for each client
        self.n_samples_train = len(train_dataset)
        if args.valid_frac > 0:
            self.n_samples_valid = len(valid_dataset)
        self.n_samples_test = len(test_dataset)

    def train_epochs(self, c_id, train_logs, args, global_params):
        """Train one client with its own data for local epochs.

        Args:
            c_id: Client ID.
            train_logs: Training logs.
            args: Other parameters for training.
            global_params: Global model parameters used in `FedProx` method.
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
        train_logs[c_id] = float(loss / n_samples)

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

    def evaluation(self, c_id, random_clients, eval_logs, args, mode="valid"):
        """Evaluation one client with its own data.

        Args:
            c_id: Client ID.
            random_clients: Randomly selected clients.
            eval_logs: Evaluation logs.
            mode: Choose valid or test mode.
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

                    if args.fed_method == "Ditto":
                        y_hat = self.per_model(x)
                    else:
                        y_hat = self.model(x)

                    pred = torch.argmax(y_hat.data, 1)

                    correct += (pred == y).sum().item()
                    n_samples += y.shape[0]

            gc.collect()
            self.acc = correct/n_samples
            if mode == "valid":
                eval_logs[c_id] = float(self.acc * self.valid_weight)
            elif mode == "test":
                eval_logs[c_id] = float(self.acc * self.test_weight)

    def get_old_eval_log(self):
        """Returns the evaluation result of the lastest round.
        """
        return self.acc

    def get_params(self):
        """Returns all of the parameters in `dict` format. Note that
        `self.params` is a reference of the model parameters.
        """
        return self.params

    def set_params(self, global_params):
        """Assigns all of the parameters with global parameters.
        """
        for name in self.params.keys():
            self.params[name].data = global_params[name].data.clone()

    def collate_fn(self, batch):
        """Moves data to `self.device`.
        """
        return tuple(x.to(self.device) for x in default_collate(batch))

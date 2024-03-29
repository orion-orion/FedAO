# -*- coding: utf-8 -*-
import math
import gc
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from utils import batch_iter


class Clients:
    def __init__(self, model_fn, args, train_datasets, valid_datasets,
                 test_datasets, data_info):
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        # Call the create function to build the computational graph of ResNet
        self.model = model_fn(self.graph)
        self.n_clients = args.n_clients

        # Initialize the model parameters
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())

        self.num_classes = data_info["num_classes"]

        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.test_datasets = test_datasets

        # Initialize the aggretation weight
        client_n_samples_train = [len(train_dataset[0])
                                  for train_dataset in self.train_datasets]
        samples_sum_train = sum(client_n_samples_train)
        self.client_train_weight = [len(train_dataset[0])/samples_sum_train
                                    for train_dataset in self.train_datasets]
        if args.valid_frac > 0:
            client_n_samples_valid = [len(valid_dataset[0])
                                      for valid_dataset in self.valid_datasets]
            samples_sum_valid = sum(client_n_samples_valid)
            self.client_valid_weight = [len(valid_dataset[0])/samples_sum_valid
                                        for valid_dataset
                                        in self.valid_datasets]
        client_n_samples_test = [len(test_dataset[0])
                                 for test_dataset in self.test_datasets]
        samples_sum_test = sum(client_n_samples_test)
        self.client_test_weight = [len(test_dataset[0])/samples_sum_test
                                   for test_dataset in self.test_datasets]

    def train_epochs(self, c_id, round, args, global_params, per=True):
        """Train one client with its own data for local epochs.

        Args:
            c_id: Client ID.
            round: Current training round.
            args: Other parameters for training.
            global_params: Global model parameters used in `FedProx` method.
            per: A flag varaible indicating whether a personalized model in
                `Ditto` method is being trained.
        """
        dataset = self.train_datasets[c_id]
        with self.graph.as_default():
            for _ in range(args.local_epochs):
                loss, n_samples = 0.0, 0
                for x, y in batch_iter(dataset, args.batch_size):
                    y = to_categorical(y, self.num_classes)
                    feed_dict = {self.model.x: x, self.model.y: y}
                    if args.fed_method == "FedProx":
                        # Multi-dimensional parameters should be flattened
                        # into one-dimensional before computing the proximal
                        # regularizer
                        feed_dict.update({self.model.global_w_vec:
                                          self.model.flatten(global_params),
                                          self.model.weight_factor: args.mu})
                    elif args.fed_method == "Ditto" and per:
                        feed_dict.update({self.model.global_w_vec:
                                          self.model.flatten(global_params),
                                          self.model.weight_factor: args.lam})

                    batch_loss, _ = self.sess.run([self.model.loss_op,
                                                   self.model.train_op],
                                                  feed_dict=feed_dict)

                    loss += batch_loss * y.shape[0]
                    n_samples += y.shape[0]

            gc.collect()
        if not (args.fed_method == "Ditto") or per:
            logging.info("Training round {}/{} - client {} -  Training Loss: "
                         "{:.3f}".format(round, args.rounds, c_id, loss
                                         / n_samples))
        return n_samples

    def evaluation(self, c_id, args, mode="valid"):
        """Evaluation one client with its own data.

        Args:
            c_id: Client ID.
            args: Evaluation arguments.
            mode: Choose valid or test mode.
        """
        if mode == "valid":
            dataset = self.valid_datasets[c_id]
        elif mode == "test":
            dataset = self.test_datasets[c_id]

        n_samples, correct = 0, 0
        with self.graph.as_default():
            for x, y in batch_iter(dataset, args.batch_size):
                y = to_categorical(y, self.num_classes)
                feed_dict = {self.model.x: x, self.model.y: y}

                logits = self.sess.run(self.model.logits,
                                       feed_dict=feed_dict)
                pred = np.argmax(logits, axis=1)

                correct += (pred == np.argmax(y, axis=1)).sum()
                n_samples += y.shape[0]

            gc.collect()
        self.acc = correct/n_samples
        return {"ACC": self.acc}

    def get_old_eval_log(self):
        """Returns the evaluation result of the lastest round.
        """
        return {"ACC": self.acc}

    def get_params(self):
        """Returns all of the parameters in `list`.
        """
        with self.graph.as_default():
            client_params = self.sess.run(tf.compat.v1.trainable_variables())
        return client_params

    def set_params(self, global_params):
        """Assigns all of the variables with global parameters. Note that
        different from the Pytorch version, `sess.run()` in tensorflow will
        get a copy of the evaluated model parameters on the CPU.
        """
        with self.graph.as_default():
            all_params = tf.compat.v1.trainable_variables()
            for variable, value in zip(all_params, global_params):
                variable.load(value, self.sess)

    def choose_clients(self, ratio=1.0):
        """Randomly chooses some clients.
        """
        choose_num = math.ceil(self.n_clients * ratio)
        return np.random.permutation(self.n_clients)[:choose_num]

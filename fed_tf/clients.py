import tensorflow as tf
import numpy as np
from collections import namedtuple
import math
import gc
import logging
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils import batch_iter


class Clients:
    def __init__(self, model_fn, args, train_datasets, valid_datasets, test_datasets, data_info):
        self.graph = tf.Graph()
        self.sess = tf.compat.v1.Session(graph=self.graph)
        # Call the create function to build the computational graph of ResNet
        self.model = model_fn(self.graph)
        self.n_clients = args.n_clients
        
        # initialize
        with self.graph.as_default():
            self.sess.run(tf.compat.v1.global_variables_initializer())

        self.num_classes = data_info["num_classes"]
        
        self.train_datasets = train_datasets
        self.valid_datasets = valid_datasets
        self.test_datasets = test_datasets

        # 初始化各client的样本占所有样本的比率
        client_n_samples_train = [len(train_dataset[0]) for train_dataset in self.train_datasets]
        samples_sum_train = sum(client_n_samples_train)
        self.client_train_prop = [len(train_dataset[0])/samples_sum_train for train_dataset in self.train_datasets]
        if args.valid_frac > 0:
            client_n_samples_valid = [len(valid_dataset[0]) for valid_dataset in self.valid_datasets]
            samples_sum_valid = sum(client_n_samples_valid)
            self.client_valid_prop = [len(valid_dataset[0])/samples_sum_valid for valid_dataset in self.valid_datasets]
        client_n_samples_test = [len(test_dataset[0])for test_dataset in self.test_datasets]
        samples_sum_test = sum(client_n_samples_test)
        self.client_test_prop = [len(test_dataset[0])/samples_sum_test for test_dataset in self.test_datasets]

    def train_epoch(self, c_id, epoch, args):
        """
            Train one client with its own data for one epoch
            c_id: Client id
            epoch: Current global epoch
            args: Training arguments
        """
        dataset = self.train_datasets[c_id]
        with self.graph.as_default():
            for _ in range(args.local_epoch):
                loss, n_samples = 0.0, 0
                for x, y in batch_iter(dataset, args.batch_size):
                    y = to_categorical(y, self.num_classes)
                    feed_dict = {self.model.x: x, self.model.y: y}
                             
                    l, _ = self.sess.run([self.model.loss_op, self.model.train_op],\
                            feed_dict=feed_dict)

                    loss += l * y.shape[0]
                    n_samples += y.shape[0]

            gc.collect()
        logging.info('Global epoch {}/{} - client {} -  Training Loss: {:.3f}'.format(epoch, args.epochs, c_id, loss / n_samples))
        return n_samples 
    
    def evaluation(self, c_id, args, mod = "valid"):
        """
            Evaluation one client with its own data for one epoch
            c_id: Client id
            args: Evaluation arguments
            mod: Choose valid or test mode
        """
        if mod == "valid":
            dataset = self.valid_datasets[c_id]
        elif mod == "test":
            dataset = self.test_datasets[c_id]

        n_samples, correct = 0, 0
        with self.graph.as_default():     
            for x, y in batch_iter(dataset, args.batch_size):
                y = to_categorical(y, self.num_classes)
                feed_dict = {self.model.x: x, self.model.y: y}

                logits = self.sess.run(self.model.logits,\
                        feed_dict=feed_dict)
                pred = np.argmax(logits, axis=1)

                correct += (pred == np.argmax(y, axis=1)).sum()
                n_samples += y.shape[0]
             
            gc.collect()
        self.acc = correct/n_samples
        return {"ACC": self.acc}

    def get_old_eval_log(self):
        """ 
            Return the evaluation result of the lastest epoch
        """
        return {"ACC": self.acc}

    def get_client_weights(self):
        """ Return all of the variables list """
        with self.graph.as_default():
            client_weights = self.sess.run(tf.compat.v1.trainable_variables())
        return client_weights

    def set_global_weights(self, global_weights):
        """ Assign all of the variables with global weights """
        with self.graph.as_default():
            # with self.graph.as_default():
            all_weights = tf.compat.v1.trainable_variables()
            for variable, value in zip(all_weights, global_weights):
                variable.load(value, self.sess)

    def choose_clients(self, ratio=1.0):
        """ randomly choose some clients """
        choose_num = math.ceil(self.n_clients * ratio)
        return np.random.permutation(self.n_clients)[:choose_num]


<p align="center">
<img src="pic/logo.png" width="500" height="190">
</p>

<div align="center">

# FedAO: A Toolbox for Federated Learning All in One

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/orion-orion/FedAO)[![](https://img.shields.io/github/license/orion-orion/FedAO)](https://github.com/orion-orion/FedAO/blob/master/LICENSE)[![](https://img.shields.io/github/stars/orion-orion/FedAO?style=social)](https://github.com/orion-orion/FedAO)
<br/>
[![](https://img.shields.io/github/directory-file-count/orion-orion/FedAO)](https://github.com/orion-orion/FedAO) [![](https://img.shields.io/github/languages/code-size/orion-orion/FedAO)](https://github.com/orion-orion/FedAO)
</div>

## 1 Introduction

[FedAO](https://github.com/orion-orion/FedAO) (Federated Learning All in One) is a toolbox for federated learning, aiming to provide implementations of [FedAvg](https://proceedings.mlr.press/v32/shamir14.html)<sup>[1]</sup>, [FedProx](https://proceedings.mlsys.org/paper_files/paper/2020/hash/1f5fe83998a09396ebe6477d9475ba0c-Abstract.html)<sup>[2]</sup>, [Ditto](https://proceedings.mlr.press/v139/li21h.html)<sup>[3]</sup>, etc. in multiple versions, such as Pytorch/Tensorflow, single-machine/distributed, synchronized/asynchronous.

If you are not familiar with distributed machine learning / federated learning, you can first read my blog [*分布式机器学习、联邦学习、多智能体的区别和联系*](https://www.cnblogs.com/orion-orion/p/15676710.html) to learn prerequisite knowledge (#\^.\^#)~. The objective function of federated learning is as follows:

$$
\begin{aligned}
    f(w) &= \sum_{k=1}^K \frac{n_k}{n} F_k(w), \\
    F_k(w) &= \frac{1}{n_k}\sum_{i = 1}^{n_k}\mathcal{l}(h(x_i; w), y_i),
\end{aligned}
$$

where $K$ is the number of clients, $n_k$ is the number of samples of the $k$-th client. The pseudocode of the [FedAvg](https://proceedings.mlr.press/v32/shamir14.html)<sup>[1]</sup>, the most basic algorithm in federated learning, is shown as follows:

<p align="center">
<img src="pic/FedAvg-pseudocode.png" width="430" height="400">
</p>

where $K$ is the number of clients, $B$ is the local minibatch size, $E$ is the number of local epochs, and $\eta$ is the learning rate.

Note that after we split the dataset according to Dirichlet distribution (first used by [TMH Hsu et al.](https://arxiv.org/abs/1909.06335)<sup>[4]</sup>) or pathological non-IID splitting (first used in the [FedAvg](https://proceedings.mlr.press/v32/shamir14.html)<sup>[1]</sup> original paper), we then split the training/validation/test set locally according to the ratio given by the user. In addition to weighting the model parameters according to the number of local training set samples in the aggregation stage, the local validation/test results of the model are also weighted according to the number of local validation/test set samples in the validation/testing stage to obtain the global validation/test results of the model.

For a detailed introduction to the dataset-splitting method in federated learning, please refer to my blog [*联邦学习：按Dirichlet分布划分Non-IID样本*](https://www.cnblogs.com/orion-orion/p/15897853.html) and [*联邦学习：按病态非独立同分布划分Non-IID样本*](https://www.cnblogs.com/orion-orion/p/15631167.html).

## 2 Dependencies

This project involves two different frameworks, Pytorch and Tensorflow. Their environmental requirements are different. You can install the corresponding Anaconda environment by yourself.

- **Pytorch** The Python version involving Pytorch code is 3.8.13, and the remaining dependencies are as follows:
  
    ```text
    numpy==1.22.3  
    tqdm
    matplotlib
    scikit-learn==1.1.1 
    pytorch==1.7.1 
    ```

- **Tensorflow** The Python version involved in the Tensorflow code is 3.8.15, and my CUDA version is 11. Because Tensorflow 1.15 only supports Python 3.7 and CUDA 10, I used the following command to install Tensorflow 1.15 on CUDA 11:

    ```bash
    pip install --upgrade pip
    pip install nvidia-pyindex
    pip install nvidia-tensorflow[horovod]
    pip install nvidia-tensorboard==1.15
    ```

    In addition to Tensorflow, the remaining dependencies are as follows:

    ```text
    numpy==1.20.0   
    tqdm
    matplotlib
    scikit-learn==1.2.0     
    ```

## 3 Dataset

This project uses the built-in datasets in Torchvision and Keras, which will be automatically downloaded and loaded in the code, without manual downloading. Pytorch code supports `EMNIST`, `FashionMNIST`, `CIFAR10`, and `CIFAR100` datasets and Tensorflow code supports `CIFAR10`, `CIFAR100` datasets (If you are in mainland China, Keras's `EMNIST` and `FashionMNIST` datasets need to be downloaded over the GFW. You can use a "ladder" or download them manually and read them offline).

The Torchvision dataset is stored in the `data` directory of the current code running path after downloaded, and the Keras dataset is stored in the `~/.keras/datasets` directory after downloaded.

The dataset can be split in two different ways: Dirichlet distribution (first used by [TMH Hsu et al.](https://arxiv.org/abs/1909.06335)<sup>[4]</sup>) and pathological non-IID splitting (first used in the [FedAvg](https://proceedings.mlr.press/v32/shamir14.html)<sup>[1]</sup> original paper).

The display of the CIFAR10 dataset splitted according to Dirichlet distribution ($\alpha=0.1$) is as follows:

<img src="pic/fed-CIFAR10-display-Dirichlet-alpha=0.1.png" width="800" height="250">

The display of the CIFAR10 dataset split according to pathological non-IID splitting (each client contains $2$ classes) is as follows:

<img src="pic/fed-CIFAR10-display-Pathological-n_shards=2.png" width="800" height="250">

## 4 Code Structure

```bash
FedAO
├── data_utils                             Data preprocessing utilities
│   ├── __init__.py                        Package initialization file
│   ├── data_split.py                      Code for splitting the dataset
│   └── plot.py                            Code for displaying the dataset
├── fed_multiprocess_syn                   Single-machine, multi-process and synchronized implementation (in Pytorch)
│   ├── client.py                          Client-side local training and validation module 
│   ├── fl.py                              The overall process of federated learning (including communication, etc.)
│   ├── main.py                            Main function, including the overall data pipeline
│   ├── model.py                           Model architecture
│   ├── server.py                          Server-side model aggregation
│   ├── subset.py                          Customized Pytorch dataset
│   └── utils.py                           Utilities for dataset loading etc.
├── fed_pytorch                            Single-machine, serial implementation (in Pytorch)
│   ├── ...
├── fed_RPC_asyn                           Distributed, asynchronous implementation (in Pytorch)
│   ├── ...
└──fed_tf                                  Single-machine, serial implementation (in Tensorflow)
    ├── ...
```

## 5 Train & Eval

### 5.1 Single-machine, serial implementation (in Pytorch)

You can first enter the corresponding path and then run `main.py` to train/validate/test the model. For example:

```bash
cd fed_pytorch    
python main.py \
        --dataset CIFAR10 \
        --n_clients 10 \
        --rounds 200 \
        --local_epochs 1 \
        --fed_method FedAvg    
```

The `--dataset` parameter is used to specify the dataset, the `--n_clients` parameter is used to specify the number of clients, `--rounds` is used to specify the number of global training rounds, and `--local_epochs` is used to specify the number of local epochs, the `--fed_method` parameter is used to specify the federated learning method used.

After the training is completed, you can view the training/validation/testing logs in the `log` directory of the code running path. In addition, the display of the dataset splitting is also stored in the `log` directory, and you can check it out.

If you need to use FedProx for training, you can use the following command:

```bash
cd fed_pytorch    
python main.py \
        --dataset CIFAR10 \
        --n_clients 10 \
        --rounds 200 \
        --local_epochs 1 \
        --fed_method FedProx \
        --mu 0.01
```

There is an additional parameter `--mu`, which represents the coefficient of the proximal regularization term in FedProx. The meaning of the remaining parameters is the same as that of the Pytorch single-machine serial implementation, and will not be described here.

If you need to use Ditto for training, you can use the following command:

```bash
cd fed_pytorch    
python main.py \
        --dataset CIFAR10 \
        --n_clients 10 \
        --rounds 200 \
        --local_epochs 1 \
        --fed_method Ditto \
        --lam 0.1
```

There is also an additional parameter `--lam`, which represents the coefficient of the proximal regularization term in Ditto. The meaning of the remaining parameters is the same as that of the Pytorch single-machine serial implementation, and will not be described here.

### 5.2 Single-machine, serial implementation (in Tensorflow)

Similarly, first, enter the corresponding path, and then run `main.py` to train/validate/test the model. For example:

```bash
cd fed_tf  
python main.py \
        --dataset CIFAR10 \
        --n_clients 10 \
        --rounds 200 \
        --local_epochs 1 \
        --fed_method FedAvg 
```

The meaning of the parameters is the same as that of the Pytorch single-machine serial implementation, and will not be described again here.

### 5.3 Single-machine, multi-process implementation (in Pytorch)

Similarly, first, enter the corresponding path, and then run `main.py` to train/validate/test the model. For example:

```bash
cd fed_multiprocess_syn
python main.py \
        --dataset CIFAR10 \
        --n_clients 10 \
        --rounds 200 \
        --local_epochs 1 \
        --fed_method FedAvg 
```

The meaning of the parameters is the same as that of the Pytorch single-machine serial implementation, and will not be described again here. However, it should be noted that in our implementation, one process corresponds to one client, so it is better not to set the number of clients too large, otherwise, it may affect the parallel efficiency and make the parallel implementation the same as the serial version.

### 5.4 Distributed, asynchronous implementation (in Pytorch)

Similarly, first, enter the corresponding path, and then run `main.py` to train/validate/test the model. For example:

```bash
cd fed_RPC_asyn
python main.py \
        --dataset CIFAR10 \
        --n_clients 10 \
        --rounds 200 \
        --local_epochs 1 \
        --lam 0.5 \
        --fed_method FedAvg
```

The meaning of the parameters here is the same as before, but there is an additional parameter $\lambda$, which is used to determine the weight corresponding to the historical model and the new model when the server updates the model. The update formula is $w^{t+1} = w^t + w_{new}$ (Refer to the asynchronous federated learning paper [*Asynchronous federated optimization*](https://arxiv.org/abs/1903.03934). The default communication domain size is `n_clients + 1`, the process with rank 0 is the master, and the rest are workers. The IP address of the master process is `localhost`, the port number is `29500`, and RPC is used to communicate between the master process and the worker processes.

## Reference

[1] McMahan B, Moore E, Ramage D, et al. Communication-efficient learning of deep networks from decentralized data[C]//Artificial intelligence and statistics. PMLR, 2017: 1273-1282.

[2] Li T, Sahu A K, Zaheer M, et al. Federated optimization in heterogeneous networks[J]. Proceedings of Machine learning and systems, 2020, 2: 429-450.

[3] Li T, Hu S, Beirami A, et al. Ditto: Fair and robust federated learning through personalization[C]//International Conference on Machine Learning. PMLR, 2021: 6357-6368.

[4] Hsu T M H, Qi H, Brown M. Measuring the effects of non-identical data distribution for federated visual classification[J]. arXiv preprint arXiv:1909.06335, 2019.

import numpy as np

def split_noniid(train_labels, alpha, n_clients):
    """
    Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha
    Args:
        train_labels: ndarray of train_labels
        alpha: the parameter of dirichlet distribution
        n_clients: number of clients
    Returns:
        client_idcs: a list containing sample idcs of clients
    """

    n_classes = train_labels.max()+1
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    # (62, 10) 记录每个client占有每个类别的多少

    class_idcs = [np.argwhere(train_labels==y).flatten() 
           for y in range(n_classes)]
    # (62, ...)，记录每个类别对应的样本下标

    
    client_idcs = [[] for _ in range(n_clients)]
    # 记录每个client对应的样本下标
    for c, fracs in zip(class_idcs, label_distribution):
        # np.split按照比例将样本划分为了对应的份数
        # for i, idcs 为遍历每一个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1]*len(c)).astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
  
    return client_idcs


def pathological_non_iid_split(train_labels, n_classes_per_client, n_clients):
    n_classes = train_labels.max()+1
    # get subset
    data_idcs = list(range(len(train_labels)))
    label2index = {k: [] for k in range(n_classes)}
    for idx in data_idcs:
        label = train_labels[idx]
        label2index[label].append(idx)

    sorted_idcs = []
    for label in label2index:
        sorted_idcs += label2index[label]

    def iid_divide(l, g):
            """
            将列表`l`分为`g`个独立同分布的group（其实就是直接划分）
            每个group都有 `int(len(l)/g)` 或者 `int(len(l)/g)+1` 个元素
            返回由不同的groups组成的列表
            """
            num_elems = len(l)
            group_size = int(len(l) / g)
            num_big_groups = num_elems - g * group_size
            num_small_groups = g - num_big_groups
            glist = []
            for i in range(num_small_groups):
                glist.append(l[group_size * i: group_size * (i + 1)])
            bi = group_size * num_small_groups
            group_size += 1
            for i in range(num_big_groups):
                glist.append(l[bi + group_size * i:bi + group_size * (i + 1)])
            return glist


    n_shards = n_clients * n_classes_per_client
        # 一共分成n_shards个独立同分布的shards
    shards = iid_divide(sorted_idcs, n_shards)
    np.random.shuffle(shards)
    # 然后再将n_shards拆分为n_client份
    tasks_shards = iid_divide(shards, n_clients)

    clients_idcs = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            # 这里shard是一个shard的数据索引(一个列表)
            # += shard 实质上是在列表里并入列表
            clients_idcs[client_id] += shard 

    return clients_idcs
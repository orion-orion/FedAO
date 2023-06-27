import numpy as np


def split_noniid(train_labels, alpha, n_clients):
    """Splits a list of data indices with corresponding labels
    into subsets according to a dirichlet distribution with parameter
    alpha.
    Args:
        train_labels: ndarray of train_labels.
        alpha: the parameter of Dirichlet distribution.
        n_clients: number of clients.
    Returns:
        client_idcs: a list containing sample idcs of clients.
    """

    n_classes = train_labels.max()+1
    # (n_classes, n_clients), label distribution matrix, indicating the
    # proportion of each label's data divided into each client
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)

    # (n_classes, ...), indicating the sample indices for each label
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    # Indicates the sample indices of each client
    client_idcs = [[] for _ in range(n_clients)]
    for c_idcs, fracs in zip(class_idcs, label_distribution):
        # `np.split` divides the sample indices of each class, i.e.`c_idcs`
        # into `n_clients` subsets according to the proportion `fracs`.
        # `i` indicates the i-th client, `idcs` indicates its sample indices
        for i, idcs in enumerate(np.split(c_idcs, (
                np.cumsum(fracs)[:-1] * len(c_idcs))
                .astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs


def pathological_non_iid_split(train_labels, n_classes_per_client, n_clients):
    n_classes = train_labels.max()+1
    data_idcs = list(range(len(train_labels)))
    label2index = {k: [] for k in range(n_classes)}
    for idx in data_idcs:
        label = train_labels[idx]
        label2index[label].append(idx)

    sorted_idcs = []
    for label in label2index:
        sorted_idcs += label2index[label]

    def iid_divide(lst, g):
        """Divides the list `l` into `g` i.i.d. groups, i.e.direct splitting.
        Each group has `int(len(l)/g)` or `int(len(l)/g)+1` elements.
        Returns a list of different groups.
        """
        num_elems = len(lst)
        group_size = int(len(lst) / g)
        num_big_groups = num_elems - g * group_size
        num_small_groups = g - num_big_groups
        glist = []
        for i in range(num_small_groups):
            glist.append(lst[group_size * i: group_size * (i + 1)])
        bi = group_size * num_small_groups
        group_size += 1
        for i in range(num_big_groups):
            glist.append(lst[bi + group_size * i:bi + group_size * (i + 1)])
        return glist

    n_shards = n_clients * n_classes_per_client
    # Divide the sample indices into `n_shards` i.i.d. shards
    shards = iid_divide(sorted_idcs, n_shards)
    np.random.shuffle(shards)
    # Then split the shards into `n_client` parts
    tasks_shards = iid_divide(shards, n_clients)

    clients_idcs = [[] for _ in range(n_clients)]
    for client_id in range(n_clients):
        for shard in tasks_shards[client_id]:
            # Here `shard` is the sample indices of a shard (a list)
            # `+= shard` is to merge the list `shard` into the list
            # `clients_idcs[client_id]`
            clients_idcs[client_id] += shard

    return clients_idcs

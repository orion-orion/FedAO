import os
import numpy as np
import matplotlib.pyplot as plt


def display_data_distribution(client_idcs, train_labels, num_classes,
                              n_clients, args):
    # Display the data distribution of each label on each client, note
    # that the x-axis of the bar chart is the client ID
    plt.figure(figsize=(20, 6))
    label_distribution = [[] for _ in range(num_classes)]
    for c_id, idc in enumerate(client_idcs):
        for idx in idc:
            label_distribution[train_labels[idx]].append(c_id)

    plt.hist(label_distribution, stacked=True,
             bins=np.arange(-0.5, n_clients + 1.5, 1),
             label=["Class {}".format(i) for i in range(num_classes)],
             rwidth=0.5)
    plt.xticks(np.arange(n_clients), ["Client {}"
               .format(c_id) for c_id in range(n_clients)])
    plt.ylabel("Number of samples")
    plt.xlabel("Client ID")
    plt.legend()
    if args.pathological_split:
        dataset_split_method = "Pathological"
        param_descr = ", n_shards=%d" % args.n_shards
    else:
        dataset_split_method = "Dirichlet"
        param_descr = "alpha={}".format(args.alpha)
    plt.title("Federated " + args.dataset + " Display ({}, {})"
              .format(dataset_split_method, param_descr))
    plot_file = os.path.join(args.log_dir, "fed-" +
                             args.dataset + "-display.png")
    plt.savefig(plot_file)

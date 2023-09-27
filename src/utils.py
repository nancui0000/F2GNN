import os
import copy
import torch
import random
import scipy
import dgl
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import torch_geometric.utils.convert as tg


def fairness_loss(pred_probs, original_labels, sens):
    """
    Calculate fairness metrics: Demographic Parity (DP) and Equal Opportunity (EO).

    Args:
    - pred_probs (torch.Tensor): The predicted probabilities tensor.
    - original_labels (torch.Tensor): The ground truth labels tensor.
    - sens (torch.Tensor): The sensitive attribute tensor.

    Returns:
    - parity (torch.Tensor): Fairness loss based on demographic parity.
    - equality (torch.Tensor): Fairness loss based on equal opportunity.
    """

    def calculate_distribution(probs, mask):
        """Compute the distribution of positive instances under a given mask."""
        N_positive = torch.sum(probs * mask)
        total_masked = torch.sum(mask.float())

        return N_positive / (total_masked + 1E-5)

    # Assume male and female masks are binary 1/0 tensors of the same size as sens
    s1_mask = sens.eq(1).float()
    s0_mask = sens.eq(0).float()

    # DP (Demographic Parity)
    s1_dist_dp = calculate_distribution(pred_probs, s1_mask)
    s0_dist_dp = calculate_distribution(pred_probs, s0_mask)
    fairness_loss_dp = torch.abs(s1_dist_dp - s0_dist_dp)

    # EO (Equal Opportunity)
    correct_probs = pred_probs * original_labels.float()
    s1_dist_eo = calculate_distribution(correct_probs, s1_mask)
    s0_dist_eo = calculate_distribution(correct_probs, s0_mask)
    fairness_loss_eo = torch.abs(s1_dist_eo - s0_dist_eo)

    return fairness_loss_dp, fairness_loss_eo


def density(n, m):
    """
    :param n: number of nodes
    :param m: number of edges
    :return: density of the graph
    """

    if m == 0 or n <= 1:
        return 0
    d = 2 * m / (n * (n - 1))
    return d


def feature_norm(features):
    feat_mean = torch.mean(features, 0)
    feat_std = torch.std(features, 0)
    return (features - feat_mean) / feat_std


def model_fair_metric(pred_y, labels, sens):
    val_y = labels
    idx_s0 = sens.eq(0)
    idx_s1 = sens.eq(1)

    idx_s0_y1 = torch.bitwise_and(idx_s0, val_y.eq(1))
    idx_s1_y1 = torch.bitwise_and(idx_s1, val_y.eq(1))

    if torch.sum(idx_s0) > 1 and torch.sum(idx_s1) > 1:
        parity = abs(torch.sum(pred_y[idx_s0]) / (torch.sum(idx_s0)) - torch.sum(pred_y[idx_s1]) / (torch.sum(idx_s1)))
    elif torch.sum(idx_s0) == 0 and torch.sum(idx_s1) > 1:
        parity = abs(torch.sum(pred_y[idx_s1]) / (torch.sum(idx_s1)))
    elif torch.sum(idx_s1) == 0 and torch.sum(idx_s0) > 1:
        parity = abs(torch.sum(pred_y[idx_s0]) / (torch.sum(idx_s0)))
    else:
        parity = 0
    if torch.sum(idx_s0_y1) > 1 and torch.sum(idx_s1_y1) > 1:
        equality = abs(
            torch.sum(pred_y[idx_s0_y1]) / (torch.sum(idx_s0_y1)) - torch.sum(pred_y[idx_s1_y1]) / (
                torch.sum(idx_s1_y1)))
    elif torch.sum(idx_s0_y1) == 0 and torch.sum(idx_s1_y1) > 1:
        equality = abs(torch.sum(pred_y[idx_s1_y1]) / (torch.sum(idx_s1_y1)))
    elif torch.sum(idx_s0_y1) > 1 and torch.sum(idx_s1_y1) == 0:
        equality = abs(torch.sum(pred_y[idx_s0_y1]) / (torch.sum(idx_s0_y1)))
    else:
        equality = 0

    return parity, equality


def load_pokec(dataset, sens_attr, predict_attr, args, path):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove("user_id")

    header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values

    sens_idx = set(np.where(sens >= 0)[0])
    label_idx = np.where(labels >= 0)[0]
    idx_used = np.asarray(list(sens_idx & set(label_idx)))

    idx_nonused = np.asarray(list(set(np.arange(len(labels))).difference(set(idx_used))))

    features = features[idx_used, :]
    labels = labels[idx_used]
    sens = sens[idx_used]

    # build graph

    idx = np.array(idx_features_labels["user_id"], dtype=np.int64)

    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=np.int64)

    idx_n = idx[idx_nonused]
    idx = idx[idx_used]

    used_ind1 = [i for i, elem in enumerate(edges_unordered[:, 0]) if elem not in idx_n]
    used_ind2 = [i for i, elem in enumerate(edges_unordered[:, 1]) if elem not in idx_n]

    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges_unordered = edges_unordered[intersect_ind, :]

    idx_map = {j: i for i, j in enumerate(idx)}

    edges_un = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=np.int64).reshape(edges_unordered.shape)

    adj = sp.coo_matrix((np.ones(edges_un.shape[0]), (edges_un[:, 0], edges_un[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    G = nx.from_scipy_sparse_array(adj)
    g_nx_ccs = (G.subgraph(c).copy() for c in nx.connected_components(G))
    g_nx = max(g_nx_ccs, key=len)

    random.seed(args.seed)
    node_ids = list(g_nx.nodes())
    idx_s = node_ids
    random.shuffle(idx_s)

    features = features[idx_s, :]

    features = features[:, np.where(np.std(np.array(features.todense()), axis=0) != 0)[0]]

    features = torch.FloatTensor(np.array(features.todense()))
    features = feature_norm(features)
    labels = torch.LongTensor(labels[idx_s])
    sens = torch.LongTensor(sens[idx_s])

    # binarize labels
    labels[labels > 1] = 1
    sens[sens > 0] = 1
    idx_map_n = {j: int(i) for i, j in enumerate(idx_s)}

    idx_nonused2 = np.asarray(list(set(np.arange(len(list(G.nodes())))).difference(set(idx_s))))
    used_ind1 = [i for i, elem in enumerate(edges_un[:, 0]) if elem not in idx_nonused2]
    used_ind2 = [i for i, elem in enumerate(edges_un[:, 1]) if elem not in idx_nonused2]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges_un = edges_un[intersect_ind, :]

    edges = np.array(list(map(idx_map_n.get, edges_un.flatten())), dtype=np.int64).reshape(edges_un.shape)

    edges = np.unique(edges, axis=0)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    degs = np.sum(adj.toarray(), axis=1) + np.ones(len(np.sum(adj.toarray(), axis=1)))

    edges = np.concatenate((np.reshape(scipy.sparse.find(adj)[0], (len(scipy.sparse.find(adj)[0]), 1)),
                            np.reshape(scipy.sparse.find(adj)[1], (len(scipy.sparse.find(adj)[1]), 1))), axis=1)

    edges = torch.LongTensor(edges.T)

    g_nx = nx.from_scipy_sparse_array(adj)

    g_dgl = dgl.from_networkx(g_nx)

    g_dgl = dgl.remove_self_loop(g_dgl)
    g_dgl = dgl.to_simple(g_dgl)
    g_dgl = dgl.add_self_loop(g_dgl)

    g_nx = g_dgl.to_networkx()
    adj = nx.adjacency_matrix(g_nx).todense()

    g_tg = tg.from_networkx(g_nx)

    print("edges.shape: ", edges.shape)

    print('num_nodes_original_graph: ', features.shape[0])
    print('num_edges_original_graph: ', edges.shape[1])
    print('sparsity__original_graph: ', density(features.shape[0], m=edges.shape[1]))

    return g_tg, g_dgl, adj, features, labels, sens, g_tg.num_nodes, g_tg.edge_index


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg


def aggregate_weights(w, norm_w):
    """
    Returns the weighted weights.
    """
    w_new = copy.deepcopy(w[0])
    for key in w_new.keys():
        temp = 0
        for i in range(0, len(w)):
            temp += torch.mul(w[i][key], norm_w[i])
        w_new[key] = temp
    return w_new


def update_weights(w_local, w_server, beta):
    """
    Returns the new local weights.
    """
    w_new = copy.deepcopy(w_local)
    for key in w_new.keys():
        w_new[key] = torch.add(torch.mul(w_new[key], (1 - beta)), torch.mul(w_server[key], beta))

    return w_new


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')

    print(f'    Local Epochs       : {args.local_ep}\n')
    return


def accuracy(output, labels):
    output = output.squeeze()
    preds = (output > 0).type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

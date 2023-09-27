import numpy as np
import scipy.sparse as sp
import torch
import dgl


class Client:
    def __init__(self, features, labels, sens, sub_nodes, sub_edges, idx_train, idx_val, idx_test, args):
        idx_intersection = list(set(idx_train.numpy()).intersection(set(sub_nodes.numpy())))
        val_idx_intersection = list(set(idx_val.numpy()).intersection(set(sub_nodes.numpy())))
        test_idx_intersection = list(set(idx_test.numpy()).intersection(set(sub_nodes.numpy())))

        idx_train_with_mask = [i for i, elem in enumerate(list(sub_nodes.numpy())) if elem in idx_intersection]
        idx_val_with_mask = [i for i, elem in enumerate(list(sub_nodes.numpy())) if elem in val_idx_intersection]
        idx_test_with_mask = [i for i, elem in enumerate(list(sub_nodes.numpy())) if elem in test_idx_intersection]

        sub_nodes = sub_nodes[sub_nodes < features.shape[0]]
        sub_features = features[sub_nodes]
        sub_labels = labels[sub_nodes]
        sub_sens = sens[sub_nodes]

        row, col = sub_edges

        sub_edges = sub_edges.T

        intra_idx = np.where((sens[row] == sens[col]) == True)[0]
        inter_idx = np.where((sens[row] != sens[col]) == True)[0]
        self.intra_ratio = len(intra_idx) / sub_edges.shape[0]
        self.inter_ratio = len(inter_idx) / sub_edges.shape[0]

        sub_nodes = np.array(sub_nodes)

        sub_idx_map = {j: i for i, j in enumerate(sub_nodes)}
        sub_edges = np.array(sub_edges, dtype=int)
        sub_edges = np.array(list(map(sub_idx_map.get, sub_edges.flatten())), dtype=int).reshape(sub_edges.shape)

        if len(sub_nodes) != 1:
            sub_adj = sp.coo_matrix((np.ones(sub_edges.shape[0]), (sub_edges[:, 0], sub_edges[:, 1])),
                                    shape=(sub_labels.shape[0], sub_labels.shape[0]), dtype=np.float32)
            sub_adj = sub_adj + sub_adj.T.multiply(sub_adj.T > sub_adj) - sub_adj.multiply(sub_adj.T > sub_adj)
            sub_adj = sub_adj + sp.eye(sub_adj.shape[0])
        else:
            sub_adj = sp.coo_matrix(np.ones((1, 1)), dtype=np.float32)

        sub_features = sp.csr_matrix(sub_features, dtype=np.float32)

        self.sub_features = torch.LongTensor(np.array(sub_features.todense()))
        self.sub_labels = torch.LongTensor(sub_labels)
        self.sub_sens = torch.LongTensor(sub_sens)

        self.sub_train_features = sub_features[idx_train_with_mask]
        self.sub_train_labels = sub_labels[idx_train_with_mask]
        self.sub_train_sens = sub_sens[idx_train_with_mask]

        self.sub_G = dgl.DGLGraph()
        self.sub_G = dgl.from_scipy(sub_adj)

        self.sub_G = dgl.remove_self_loop(self.sub_G)
        self.sub_G = dgl.to_simple(self.sub_G)
        self.sub_G = dgl.add_self_loop(self.sub_G)

        self.sub_adj = sub_adj
        self.sub_edges = sub_edges
        self.num_sub_edges = sub_edges.shape[0]
        self.idx_train_with_mask = idx_train_with_mask
        self.idx_val_with_mask = idx_val_with_mask
        self.idx_test_with_mask = idx_test_with_mask


class Subgraph:
    def __init__(self, idx, features, labels, sens, edges, args):
        features = features[idx]
        labels = labels[idx]
        sens = sens[idx]

        if edges.shape[0] <= 2:
            edges = edges.T
        idx = np.array(idx)
        idx_map = {j: i for i, j in enumerate(idx)}
        edges = np.array(edges, dtype=int)
        edges = np.array(list(map(idx_map.get, edges.flatten())), dtype=int).reshape(edges.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.size, labels.size), dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = adj + sp.eye(adj.shape[0])

        features = sp.csr_matrix(features, dtype=np.float32)

        self.features = torch.LongTensor(np.array(features.todense()))
        self.labels = torch.LongTensor(labels)
        self.sens = torch.LongTensor(sens)
        self.G = dgl.from_scipy(adj)
        self.adj = adj
        self.edges = edges

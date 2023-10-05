import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
from options import args_parser
from models import FairGNN
from utils import average_weights, aggregate_weights, exp_details, load_pokec, accuracy, density, update_weights, \
    model_fair_metric

from sklearn.metrics import roc_auc_score, recall_score, f1_score
import random
from terminal import Client, Subgraph
from torch_geometric.utils import subgraph, k_hop_subgraph
from time import perf_counter as t
from scipy.special import softmax

from scipy.spatial import distance

from collections import Counter
import warnings

torch.set_printoptions(profile="full")
torch.autograd.set_detect_anomaly(True)

warnings.filterwarnings("ignore", category=UserWarning)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.gpu != 'cpu':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def fair_metric(output, idx, labels, sens):
    val_y = labels[idx].cpu().numpy()
    idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()] == 0
    idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()] == 1

    idx_s0_y1 = np.bitwise_and(idx_s0, val_y == 1)
    idx_s1_y1 = np.bitwise_and(idx_s1, val_y == 1)

    pred_y = (output[idx].squeeze() > 0).type_as(labels).cpu().numpy()
    parity = abs(sum(pred_y[idx_s0]) / sum(idx_s0) - sum(pred_y[idx_s1]) / sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1]) / sum(idx_s0_y1) - sum(pred_y[idx_s1_y1]) / sum(idx_s1_y1))

    return parity, equality


def local_fairness(model, clients, device):
    local_parity, local_equality = [], []
    model = model.to(device)
    model.eval()
    for client in clients:
        model.eval()
        client.sub_G = client.sub_G.to(device)
        client.sub_features = client.sub_features.to(device)
        local_ouput = model(client.sub_G, client.sub_features)
        local_parity_element, local_equality_element = fair_metric(local_ouput, client.idx_train_with_mask,
                                                                   client.sub_labels, client.sub_sens)
        local_parity.append(local_parity_element)
        local_equality.append(local_equality_element)

    mean_local_parity = sum(local_parity) / len(local_parity)
    mean_local_equality = sum(local_equality) / len(local_equality)
    print("mean_local_parity: ", mean_local_parity.item())
    print("mean_local_equality: ", mean_local_equality.item())
    return mean_local_parity.item(), mean_local_equality.item()


if __name__ == '__main__':

    args = args_parser()
    exp_details(args)

    path_project = os.path.abspath('..')
    logger = SummaryWriter(log_dir=args.log_dir)

    if args.gpu != 'cpu':
        # torch.cuda.set_device(int(args.gpu))
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True)
        # device = 'cuda:' + str(args.gpu)
        device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = 'cpu'
    print("device: ", device)

    set_seed(args.seed)

    # load dataset and user groups

    if args.dataset == 'pokec-z':
        dataset = 'region_job'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"

        path = "../data/pokec/"
        test_idx = False

        graph, G, adj, features, labels, sens, num_nodes, edge_index = load_pokec(dataset, sens_attr, predict_attr,
                                                                                  args,
                                                                                  path=path)
    elif args.dataset == 'pokec-n':
        dataset = 'region_job_2'
        sens_attr = "region"
        predict_attr = "I_am_working_in_field"

        path = "../data/pokec/"
        test_idx = False

        graph, G, adj, features, labels, sens, num_nodes, edge_index = load_pokec(dataset, sens_attr, predict_attr,
                                                                                  args,
                                                                                  path=path)

    elif args.dataset == 'NBA':
        dataset = 'nba'
        sens_attr = 'country'
        predict_attr = 'SALARY'
        path = '../dataset/NBA'

        graph, G, adj, features, labels, sens, num_nodes, edge_index = load_pokec(dataset, sens_attr, predict_attr,
                                                                                  args,
                                                                                  path=path)

    label_idx = np.where(labels >= 0)[0]
    sens_idx = set(np.where(sens >= 0)[0])
    idx_used = np.asarray(list(sens_idx & set(label_idx)))
    random.seed(args.seed)
    random.shuffle(idx_used)
    idx_train = idx_used[:int(0.5 * len(idx_used))]
    idx_train = torch.LongTensor(idx_train)
    idx_val = idx_used[int(0.5 * len(idx_used)):int(0.75 * len(idx_used))]
    idx_val = torch.LongTensor(idx_val)
    idx_test = idx_used[int(0.75 * len(idx_used)):]
    idx_test = torch.LongTensor(idx_test)

    edge_index = torch.LongTensor(graph.edge_index)

    edges_train = subgraph(idx_train, edge_index)[0]
    edges_val = subgraph(idx_val, edge_index)[0]
    edges_test = subgraph(idx_test, edge_index)[0]

    idx_used = torch.LongTensor(idx_used)

    print('num_nodes_applied_graph: ', len(idx_used))
    print('num_nodes_test_graph: ', features[idx_test].shape[0])
    print('num_nodes_train_graph: ', features[idx_train].shape[0])
    print('num_edges_test_graph: ', edges_test.shape[1])
    print('sparsity_test_graph: ', density(features[idx_test].shape[0], m=edges_test.shape[1]))

    clients = []
    temp = []
    density_ego_networks = []
    num_nodes_per_client = []
    intra_ratio_per_client = []
    inter_ratio_per_client = []

    print("Begin to generate ego networks.")
    random.seed(args.seed)
    for i in range(args.ego_number):
        sub_nodes = torch.LongTensor([1, 1])
        while sub_nodes.shape[0] <= 2:
            ego = random.choice(idx_train.tolist())
            sub_nodes, sub_edges, _, _ = k_hop_subgraph(node_idx=ego, num_hops=args.num_hops,
                                                        edge_index=edge_index)
        clients.append(Client(features, labels, sens, sub_nodes, sub_edges, idx_train, idx_val, idx_test, args))
        num_nodes_per_client.append(float(sub_nodes.shape[0]))

        intra_ratio_per_client.append(clients[-1].intra_ratio)
        inter_ratio_per_client.append(clients[-1].inter_ratio)

        temp.extend(sub_nodes.cpu().numpy().tolist())

    weight_num_nodes_per_client = np.array([float(i) / sum(num_nodes_per_client) for i in num_nodes_per_client])

    weight_fairfed = weight_num_nodes_per_client
    # print("temp: ", temp)

    nodes_unique = np.unique(temp)

    # use Counter to get count of each element
    count_dict = Counter(temp)

    # filter the elements that appear more than once
    duplicates = {item: count for item, count in count_dict.items() if count > 1}

    print('num_nodes_overlap: ', len(duplicates))
    print('overlap partation: ', len(duplicates) / len(idx_used))

    # BUILD MODEL

    if args.model in ["GCN", "SGC"]:
        local_models = []
        for i in range(args.ego_number):
            local_models.append(FairGNN(nfeat=features.shape[1], args=args))

        global_model = FairGNN(nfeat=features.shape[1], args=args)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    if args.gpu:
        global_model = global_model.to(device)
        features = torch.LongTensor(np.array(features)).to(device)
        labels = torch.LongTensor(labels).to(device)
        idx_train = torch.LongTensor(idx_train).to(device)
        idx_val = idx_val.to(device)
        idx_test = torch.LongTensor(idx_test).to(device)
        sens = torch.LongTensor(sens).to(device)
        G = G.to(device)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, train_accuracy = [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 2
    val_loss_pre, counter = 0, 0
    best_result = {}
    cls_best_result = {}
    best_fair = 100
    acc_old = 0
    best_fair = 999
    acc_counter = 0
    counter = 0
    patience = args.patience
    inter_intra_ratio = []

    for i in range(len(clients)):
        local_models[i] = local_models[i].to(device)

    start = t()
    np.random.seed(args.seed)
    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses, local_cls_losses, local_acc = [], [], [], []

        local_parity, local_equality = [], []

        print(f'\n | Training Round : {epoch + 1} |\n')

        if args.ego_number <= 200:
            egos = np.random.choice(range(len(clients)), int(args.ego_number / 10))
        elif args.ego_number == 1000:
            egos = np.random.choice(range(len(clients)), int(args.ego_number / 50))
        else:
            egos = np.random.choice(range(len(clients)), int(args.ego_number / 20))

        for i in egos:
            # print('i: ', i)
            if args.gpu:
                clients[i].sub_features = clients[i].sub_features.to(device)
                clients[i].sub_labels = clients[i].sub_labels.to(device)
                clients[i].sub_sens = clients[i].sub_sens.to(device)
                clients[i].sub_G = clients[i].sub_G.to(device)
                local_models[i] = local_models[i].to(device)

            local_eo = []

            if not clients[i].sub_labels.shape:
                clients[i].sub_labels = torch.LongTensor([clients[i].sub_labels]).to(device)
            tau = 1
            # with torch.autograd.detect_anomaly():
            if args.update_method == 'combine':

                y_global = global_model(clients[i].sub_G, clients[i].sub_features)
                y_global_ = y_global.squeeze()
                y_global_ = y_global[clients[i].idx_train_with_mask]

                y_local = local_models[i](clients[i].sub_G, clients[i].sub_features)
                y_local_ = y_local.squeeze()
                y_local_ = y_local[clients[i].idx_train_with_mask]

                if len(y_global_) >= 2 and len(y_local_) >= 2:

                    preds_y_global = (y_global_ > 0).type_as(labels)
                    temp_y_global = [float(i) / args.tau_combine for i in preds_y_global]
                    y_global = softmax(temp_y_global)

                    preds_y_local = (y_local_ > 0).type_as(labels)
                    temp_y_local = [float(i) / args.tau_combine for i in preds_y_local]
                    y_local = softmax(temp_y_local)

                    Beta = distance.jensenshannon(y_global, y_local)
                elif len(y_global_) == 1 and len(y_local_) == 1:
                    y_global_ = y_global_.detach().cpu().numpy()
                    y_global_[y_global_ > 0] = 1
                    temp_y_global = y_global_ / args.tau_combine
                    y_global = softmax(temp_y_global)

                    y_local_ = y_local_.detach().cpu().numpy()
                    y_local_[y_local_ > 0] = 1
                    temp_y_local = y_local_ / args.tau_combine
                    y_local = softmax(temp_y_local)

                    Beta = distance.jensenshannon(y_global, y_local)
                    Beta = torch.FloatTensor(Beta)
                    Beta = Beta.to(device)
                else:
                    pass

            if args.update_method == 'combine':
                if len(y_global_) >= 1 and len(y_local_) >= 1:
                    new_local_weights = update_weights(local_models[i].state_dict(), global_weights, Beta)
                    local_models[i].load_state_dict(new_local_weights)
                else:
                    local_models[i].load_state_dict(copy.deepcopy(global_weights))
            elif args.update_method == 'uniform':
                local_models[i].load_state_dict(copy.deepcopy(global_weights))

            for j in range(args.local_ep):
                local_models[i].train()
                local_models[i].optimize(clients[i].sub_G, clients[i].sub_features, clients[i].sub_labels,
                                         clients[i].sub_sens, clients[i].idx_train_with_mask)

            local_weights.append(copy.deepcopy(local_models[i].state_dict()))
            local_losses.append(local_models[i].G_loss.item())
            local_cls_losses.append(local_models[i].cls_loss.item())

            local_parity.append(local_models[i].parity)
            local_equality.append(local_models[i].equality)

            local_acc.append(local_models[i].acc)

            local_eo.append(local_models[i].equality)

        reduction_functions = {'median': torch.median, 'mean': torch.mean, 'min': torch.min}

        if args.local_fairness_measure in reduction_functions:
            reduction_func = reduction_functions[args.local_fairness_measure]

            input_local_parity = torch.cat([
                torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(device)
                if isinstance(t, int) else t.unsqueeze(0).to(device)
                for t in local_parity
            ])
            input_local_equality = torch.cat([
                torch.tensor(t, dtype=torch.float32).unsqueeze(0).to(device)
                if isinstance(t, int) else t.unsqueeze(0).to(device)
                for t in local_equality
            ])

            measure_local_parity = reduction_func(input_local_parity)
            measure_local_equality = reduction_func(input_local_equality)

        tags_local_update = ["train_loss", "local_cls_loss", "local_train_parity_median", "local_train_equality_median"]
        logger.add_scalar(tags_local_update[0], local_losses[-1], epoch)
        logger.add_scalar(tags_local_update[1], local_cls_losses[-1], epoch)
        logger.add_scalar(tags_local_update[2], measure_local_parity.item(), epoch)
        logger.add_scalar(tags_local_update[3], measure_local_equality.item(), epoch)

        # update global weights

        if args.aggregate_method == 'avg':
            global_weights = average_weights(local_weights)
        else:

            if args.aggregate_method == '1-|inter-intra|':
                difference_weight_vector = [abs(inter_ratio_per_client[i] - intra_ratio_per_client[i]) for i in
                                            egos]
                difference_weight_vector2 = [(1 - i) for i in difference_weight_vector]

                temp_weights_vector = softmax(difference_weight_vector2)

                temp_weights_vector2 = [(1 / (local_models[i].equality + local_models[i].parity + 1E-5)) for i in
                                        egos]
                temp_weights_vector2 = torch.tensor(temp_weights_vector2, device=device)
                temp_weights_vector2 = torch.softmax(temp_weights_vector2, dim=-1)

                temp_weights_vector2 = [torch.exp(i) for i in temp_weights_vector2]

                normalized_weights_vector = [(args.lambda1 * w1 + w2) / args.tau
                                             for w1, w2 in zip(temp_weights_vector, temp_weights_vector2)]
                normalized_weights_vector = torch.tensor(normalized_weights_vector, device=device)
                normalized_weights_vector = torch.softmax(normalized_weights_vector, dim=-1)
            else:
                print("args.aggregate_method is wrong!")
            global_weights = aggregate_weights(local_weights, normalized_weights_vector)

        # update global weights

        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        train_acc_avg = sum(local_acc) / len(local_acc)

        list_acc, list_roc, list_parity, list_equality = [], [], [], []

        global_model.eval()

        output = global_model(G, features)

        acc_val = accuracy(output[idx_val], labels[idx_val])
        list_acc.append(acc_val)

        roc_val = roc_auc_score(np.float64(labels[idx_val].detach().cpu().numpy()),
                                np.float64(output[idx_val].detach().cpu().numpy()))
        list_roc.append(roc_val)

        parity_val, equality_val = fair_metric(output, idx_val, labels, sens)

        acc_test = accuracy(output[idx_test], labels[idx_test])
        roc_test = roc_auc_score(np.float64(labels[idx_test].detach().cpu().numpy()),
                                 np.float64(output[idx_test].detach().cpu().numpy()))
        parity, equality = fair_metric(output, idx_test, labels, sens)

        list_test_parity, list_test_equality = [], []
        list_test_acc, list_test_roc = [], []
        for i in range(args.ego_number):
            client = clients[i]
            global_model.eval()
            output = global_model(G, features)
            pred_local_fairness_part = output[client.idx_test_with_mask]
            labels_local_fairness_part = labels[client.idx_test_with_mask]

            pred_local_fairness_part = pred_local_fairness_part.squeeze()
            pred_local_fairness_part = (pred_local_fairness_part > 0).type_as(labels_local_fairness_part)

            acc_llocal = accuracy(output[client.idx_test_with_mask], labels[client.idx_test_with_mask])
            try:
                roc_llocal = roc_auc_score(np.float64(labels[client.idx_test_with_mask].detach().cpu().numpy()),
                                           np.float64(output[client.idx_test_with_mask].detach().cpu().numpy()))
                list_test_roc.append(roc_llocal)
            except ValueError:
                pass
            except TypeError:
                pass

            sens_local_fairness_part = sens[client.idx_test_with_mask]

            parity_llocal, equality_llocal = model_fair_metric(pred_local_fairness_part, labels_local_fairness_part,
                                                               sens_local_fairness_part)

            list_test_parity.append(parity_llocal)
            list_test_equality.append(equality_llocal)
            list_test_acc.append(acc_llocal)

        reduction_functions = {'median': torch.median, 'mean': torch.mean, 'min': torch.min}

        if args.local_fairness_measure in reduction_functions:
            reduction_func = reduction_functions[args.local_fairness_measure]
            measure_local_test_equality = reduction_func(
                torch.cat([torch.tensor(t, dtype=torch.float32).to(device).unsqueeze(0) for t in list_test_equality]))
            measure_local_test_parity = reduction_func(
                torch.cat([torch.tensor(t, dtype=torch.float32).to(device).unsqueeze(0) for t in list_test_parity]))

            measure_local_test_acc = reduction_func(
                torch.cat([torch.tensor(t, dtype=torch.float32).to(device).unsqueeze(0) for t in list_test_acc]))
            measure_local_test_roc = reduction_func(
                torch.cat([torch.tensor(t, dtype=torch.float32).to(device).unsqueeze(0) for t in list_test_roc]))

        if float(acc_old) < float(acc_val):
            cls_best_result['acc'] = acc_val.item()
            cls_best_result['roc'] = roc_val
            cls_best_result['parity'] = parity_val
            cls_best_result['equality'] = equality_val

            cls_best_result['acc_test'] = acc_test.item()
            cls_best_result['roc_test'] = roc_test
            cls_best_result['parity_test'] = parity
            cls_best_result['equality_test'] = equality
            cls_best_result['measure_local_test_parity'] = measure_local_test_parity.item()
            cls_best_result['measure_local_test_equality'] = measure_local_test_equality.item()
            cls_best_result['measure_local_test_acc'] = measure_local_test_acc.item()
            cls_best_result['measure_local_test_roc'] = measure_local_test_roc.item()
            cls_best_result['epoch'] = epoch + 1
            cls_best_model = global_model
            acc_old = acc_val

            best_result['acc'] = acc_val.item()
            best_result['roc'] = roc_val
            best_result['parity'] = parity_val
            best_result['equality'] = equality_val

            best_result['acc_test'] = acc_test.item()
            best_result['roc_test'] = roc_test
            best_result['parity_test'] = parity
            best_result['equality_test'] = equality
            best_result['measure_local_test_parity'] = measure_local_test_parity.item()
            best_result['measure_local_test_equality'] = measure_local_test_equality.item()
            best_result['measure_local_test_acc'] = measure_local_test_acc.item()
            best_result['measure_local_test_roc'] = measure_local_test_roc.item()
            best_result['epoch'] = epoch + 1

        if acc_val > args.acc:
            if best_fair > parity_val + equality_val:
                best_result['acc'] = acc_val.item()
                best_result['roc'] = roc_val
                best_result['parity'] = parity_val
                best_result['equality'] = equality_val

                best_result['acc_test'] = acc_test.item()
                best_result['roc_test'] = roc_test
                best_result['parity_test'] = parity
                best_result['equality_test'] = equality
                best_result['measure_local_test_parity'] = measure_local_test_parity.item()
                best_result['measure_local_test_equality'] = measure_local_test_equality.item()
                best_result['measure_local_test_acc'] = measure_local_test_acc.item()
                best_result['measure_local_test_roc'] = measure_local_test_roc.item()
                best_result['epoch'] = epoch + 1
                best_fair = parity_val + equality_val
                best_model = global_model

        if np.abs(float(acc_val) - float(acc_counter)) <= 1E-5:
            counter += 1
        else:
            counter = 0

        acc_counter = float(acc_val)

        if counter > patience and epoch >= 250:

            break

        tags_global_update = ["val_acc", "val_roc", "val_parity", "val_equality", "test_acc", "test_roc", "test_parity",
                              "test_equality", 'measure_local_test_parity', 'measure_local_test_equality']
        logger.add_scalar(tags_global_update[0], acc_val.item(), epoch + 1)
        logger.add_scalar(tags_global_update[1], roc_val, epoch + 1)
        logger.add_scalar(tags_global_update[2], parity_val, epoch + 1)
        logger.add_scalar(tags_global_update[3], equality_val, epoch + 1)
        logger.add_scalar(tags_global_update[4], acc_test.item(), epoch + 1)
        logger.add_scalar(tags_global_update[5], roc_test, epoch + 1)
        logger.add_scalar(tags_global_update[6], parity, epoch + 1)
        logger.add_scalar(tags_global_update[7], equality, epoch + 1)
        logger.add_scalar(tags_global_update[8], measure_local_test_parity, epoch + 1)
        logger.add_scalar(tags_global_update[9], measure_local_test_equality, epoch + 1)

        print("\n=================================")
        print("counter: ", counter)
        print('Epoch: {:04d}\n'.format(epoch + 1),
              '*acc_train*: {:.4f}\n'.format(train_acc_avg),
              '*acc_val*: {:.4f}'.format(acc_val.item()),
              "auc_val: {:.4f}".format(roc_val),
              "parity_val: {:.4f}".format(parity_val),
              "equality: {:.4f}".format(equality_val),
              "measure_local_test_parity: {:.4f}".format(measure_local_test_parity),
              "measure_local_test_equality: {:.4f}".format(measure_local_test_equality))
        print("Test:\n",
              "*acc_test*: {:.4f}".format(acc_test.item()),
              "auc: {:.4f}".format(roc_test),
              "parity: {:.4f}".format(parity),
              "equality: {:.4f}".format(equality),
              "measure_local_test_parity: {:.4f}".format(measure_local_test_parity),
              "measure_local_test_equality: {:.4f}".format(measure_local_test_equality))

    now = t()

    if len(best_result) > 4:
        print("Classifying Best Results:\n",
              "Epoch: ", cls_best_result['epoch'], "\n",
              "Validition accuracy: {:.4f}".format(cls_best_result['acc']),
              "Validition roc: {:.4f}".format(cls_best_result['roc']),
              "Validition parity: {:.4f}".format(cls_best_result['parity']),
              "Validition equality: {:.4f}".format(cls_best_result['equality']),
              "\n",
              "Test accuracy: {:.4f}".format(cls_best_result['acc_test']),
              "Test roc: {:.4f}".format(cls_best_result['roc_test']),
              "Test parity: {:.4f}".format(cls_best_result['parity_test']),
              "Test equality: {:.4f}".format(cls_best_result['equality_test']),
              "Test measure_local_parity: {:.4f}".format(cls_best_result['measure_local_test_parity']),
              "Test measure_local_equality: {:.4f}".format(cls_best_result['measure_local_test_equality']),
              "Test measure_local_acc: {:.4f}".format(cls_best_result['measure_local_test_acc']),
              "Test measure_local_roc: {:.4f}".format(cls_best_result['measure_local_test_roc']),
              "\n",
              "Best Results:\n",
              "Epoch: ", best_result['epoch'], "\n",
              "Validition accuracy: {:.4f}".format(best_result['acc']),
              "Validition roc: {:.4f}".format(best_result['roc']),
              "Validition parity: {:.4f}".format(best_result['parity']),
              "Validition equality: {:.4f}".format(best_result['equality']),
              "\n"
              "Test accuracy: {:.4f}".format(best_result['acc_test']),
              "Test roc: {:.4f}".format(best_result['roc_test']),
              "Test parity: {:.4f}".format(best_result['parity_test']),
              "Test equality: {:.4f}".format(best_result['equality_test']),
              "Test measure_local_parity: {:.4f}".format(best_result['measure_local_test_parity']),
              "Test measure_local_equality: {:.4f}".format(best_result['measure_local_test_equality']),
              "\n"
              "Test measure_local_acc: {:.4f}".format(best_result['measure_local_test_acc']),
              "Test measure_local_roc: {:.4f}".format(best_result['measure_local_test_roc']),
              "\n"
              "Time: {:.4f} mins".format((now - start) / 60),
              "\n",
              "Dataset: ", args.dataset,
              "\nClients Number: ", args.ego_number,
              "\nUpdate_Method: ", args.update_method,
              "\nAggregate Method: ", args.aggregate_method,
              "\nAlpha: ", args.alpha,
              "\nlambda1: ", args.lambda1,
              "\ntau: ", args.tau,
              "\npenalty: ", args.penalty,
              "\nseed: ", args.seed
              )

    else:
        print("Classifying Best Results:\n",
              "Epoch: ", cls_best_result['epoch'], "\n",
              "Validition accuracy: {:.4f}".format(cls_best_result['acc']),
              "Validition roc: {:.4f}".format(cls_best_result['roc']),
              "Validition parity: {:.4f}".format(cls_best_result['parity']),
              "Validition equality: {:.4f}".format(cls_best_result['equality']),
              "\n"
              "Test accuracy: {:.4f}".format(cls_best_result['acc_test']),
              "Test roc: {:.4f}".format(cls_best_result['roc_test']),
              "Test parity: {:.4f}".format(cls_best_result['parity_test']),
              "Test equality: {:.4f}".format(cls_best_result['equality_test']),
              "Test measure_local_parity: {:.4f}".format(cls_best_result['measure_local_test_parity']),
              "Test measure_local_equality: {:.4f}".format(cls_best_result['measure_local_test_equality']),
              "Test measure_local_acc: {:.4f}".format(cls_best_result['measure_local_test_acc']),
              "Test measure_local_roc: {:.4f}".format(cls_best_result['measure_local_test_roc']))

    logger.close()

    # exp_log = f'exp_log_{args.dataset}_{args.ego_number}_clients_{args.num_hops}_hop_{args.model}.txt'

    with open("exp_log.txt", mode='a+') as file:
        file.write("\n=================================================\n")
        file.write(
            "Parameters: \n" + str(args)
        )
        if len(best_result) > 4:
            file.write(
                "\n" +
                "Best Results:\n" +
                "Epoch: " + str(best_result['epoch']) + "\n" +
                "Validation accuracy: " + str(best_result['acc']) +
                " Validation auc: " + str(best_result['roc']) +
                " Validation parity: " + str(best_result['parity']) +
                " Validation equality: " + str(best_result['equality']) +
                "\n" +
                "Test accuracy: " + str(best_result['acc_test']) +
                " Test auc: " + str(best_result['roc_test']) +
                " Test parity: " + str(best_result['parity_test']) +
                " Test equality: " + str(best_result['equality_test']) +
                " Test measure_local_parity: " + str(best_result['measure_local_test_parity']) +
                " Test measure_local_equality: " + str(best_result['measure_local_test_equality']) +
                " Test measure_local_acc: " + str(best_result['measure_local_test_acc']) +
                " Test measure_local_roc: " + str(best_result['measure_local_test_roc']) +
                "\n" +
                "The Best Classifying Results:\n" +
                "Test accuracy: " + str(cls_best_result['acc_test']) +
                " Test auc: " + str(cls_best_result['roc_test']) +
                " Test parity: " + str(cls_best_result['parity_test']) +
                " Test equality: " + str(cls_best_result['equality_test']) +
                " Test measure_local_parity: " + str(cls_best_result['measure_local_test_parity']) +
                " Test measure_local_equality: " + str(cls_best_result['measure_local_test_equality']) +
                " Test measure_local_acc: " + str(cls_best_result['measure_local_test_acc']) +
                " Test measure_local_roc: " + str(cls_best_result['measure_local_test_roc']) +
                "\n" +
                "Time: " + str((now - start) / 60) + "mins"
            )
        else:
            file.write(
                "\n" +
                "The Best Classifying Results:\n" +
                "Test accuracy: " + str(cls_best_result['acc_test']) +
                " Test auc: " + str(cls_best_result['roc_test']) +
                " Test parity: " + str(cls_best_result['parity_test']) +
                " Test equality: " + str(cls_best_result['equality_test']) +
                " ")
    file.close()

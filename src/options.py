import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=500,
                        help="number of rounds of training")
    parser.add_argument('--ego_number', type=int, default=10, help='the number of egonetworks')
    parser.add_argument('--local_ep', type=int, default=10,
                        help="the number of local epochs: E")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')

    # model arguments
    parser.add_argument('--model', type=str, default='GCN', help='model name')

    # GNN model arguments
    parser.add_argument('--num_hidden', type=int, default=128,
                        help='Number of hidden units of classifier.')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--alpha', type=float, default=4, help='The hyperparameter of alpha')

    parser.add_argument('--acc', type=float, default=0.66,
                        help='the selected FairGNN accuracy on val would be at least this high')

    parser.add_argument('--num_hops', type=int, default=4, help='the number hops of each egonetwork')

    # other arguments
    parser.add_argument('--dataset', type=str, default='pokec-z', help="name \
                        of dataset")

    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optimizer', type=str, default='adam', help="type \
                        of optimizer")

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--update_method', type=str, default='combine', help='how to update local models')
    parser.add_argument('--aggregate_method', type=str, default='1-|inter-intra|',
                        help='how to aggregate local models and build '
                             'global model')
    parser.add_argument('--patience', type=int, default='30', help='early stopping threshold')
    parser.add_argument('--penalty', type=str, default='fair', help='regulization item type in the local model')
    parser.add_argument('--tau', type=float, default='1', help='temperature parameter in server aggregation')
    parser.add_argument('--tau_combine', type=float, default='0.001', help='temperature parameter in local '
                                                                           'initailized update')
    parser.add_argument('--tau_fair_loss', type=float, default='4', help='temperature parameter in local')
    parser.add_argument('--lambda1', type=float, default='1', help='The hyperparameter to weight edges balance')

    parser.add_argument('--log_dir', type=str, default='../logs', help='directory for logs')

    parser.add_argument('--local_fairness_measure', type=str, default='median', help='local fairness measure')

    args = parser.parse_args()
    return args

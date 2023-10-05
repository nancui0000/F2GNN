
import torch
import torch.nn.functional as F
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn import SGConv
from utils import fairness_loss


def get_model(nfeat, args):
    if args.model == "GCN":
        model = GCN_Body(nfeat, args.num_hidden, args.dropout)
    elif args.model == "SGC":
        model = SGC_Body(nfeat, args.num_hidden, args.dropout)
    else:
        print("Model not implement")
        return

    return model


class FairGNN(nn.Module):

    def __init__(self, nfeat, args):
        super(FairGNN, self).__init__()

        nhid = args.num_hidden
        dropout = args.dropout
        self.device = 'cuda' if args.gpu else 'cpu'
        self.GNN = get_model(nfeat, args)
        self.classifier = nn.Linear(nhid, 1)

        self.G_params = list(self.GNN.parameters()) + list(self.classifier.parameters())
        self.optimizer_G = torch.optim.Adam(self.G_params, lr=args.lr, weight_decay=args.weight_decay)

        self.args = args

        self.criterion = nn.BCEWithLogitsLoss()

        self.G_loss = 0

    def forward(self, g, x):
        z = self.GNN(g, x)
        y = self.classifier(z)

        return y

    def optimize(self, g, x, labels, sens, idx_train_with_mask):
        self.optimizer_G.zero_grad()
        h = self.GNN(g, x)
        y = self.classifier(h)
        y_train = y[idx_train_with_mask]
        labels_train = labels[idx_train_with_mask]
        y_for_loss = torch.sigmoid(y_train / self.args.tau_fair_loss)

        y_output = y_train.squeeze()
        preds = (y_output > 0).type_as(labels_train)
        correct = preds.eq(labels_train).double()
        correct = correct.sum()
        self.acc = correct / len(labels_train)

        self.parity, self.equality = fairness_loss(y_for_loss, labels_train, sens[idx_train_with_mask])

        self.cls_loss = self.criterion(y_train, labels_train.unsqueeze(1).float())

        if self.args.penalty == 'fair':

            self.G_loss = self.cls_loss + self.args.alpha * (self.parity + self.equality)

        else:
            self.G_loss = self.cls_loss

        self.G_loss.backward()

        self.optimizer_G.step()


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()
        self.body = GCN_Body(nfeat, nhid, dropout)
        self.fc = nn.Linear(nhid, nclass)

    def forward(self, g, x):
        x = self.body(g, x)
        x = self.fc(x)
        return x


class GCN_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(GCN_Body, self).__init__()
        self.gc1 = GraphConv(nfeat, nhid)
        self.gc2 = GraphConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = F.relu(self.gc1(g, x))
        x = self.dropout(x)

        x = self.gc2(g, x)

        return x


class SGC_Body(nn.Module):
    def __init__(self, nfeat, nhid, dropout):
        super(SGC_Body, self).__init__()

        self.sgc1 = SGConv(nfeat, nhid)
        self.sgc2 = SGConv(nhid, nhid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g, x):
        x = self.sgc1(g, x)
        x = self.dropout(x)
        x = self.sgc2(g, x)

        return x

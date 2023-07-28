from dgl.nn.pytorch import SAGEConv
from dgl.data import CoraGraphDataset
import torch.nn.functional as F
import torch.nn as nn
import torch


class GraphSAGE(nn.Module):  # GraphSAGE model
    def __init__(self, graph, in_feats, hid_feats, out_feats, nums_layers, activation, prob_dropout):
        super(GraphSAGE, self).__init__()
        self.graph = graph
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, hid_feats, activation=activation, aggregator_type='mean'))
        for _ in range(nums_layers - 1):
            self.layers.append(SAGEConv(hid_feats, hid_feats, activation=activation, aggregator_type='mean'))
        self.layers.append(SAGEConv(hid_feats, out_feats, activation=None, aggregator_type='mean'))
        self.dropout = nn.Dropout(p=prob_dropout)

    def forward(self, features):
        h = features
        for i, layers in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layers(self.graph, h)
        return h


def evaluate(model, features, labels, mask):  # model evaluation
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        indices = torch.max(logits, dim=1)[1]
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


# cora dataset attributions
dataset = CoraGraphDataset()
g = dataset[0]
feats = g.ndata['feat']
labels = g.ndata['label']
train_mask = g.ndata['train_mask']
valid_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
n_class = dataset.num_classes
n_feat = feats.shape[1]

# hyperparameters
dropout = 0.5
lr = 0.01
n_epochs = 200
n_hidden = 100
n_layer = 2
weight_decay = 5e-4

# model
model = GraphSAGE(graph=g, in_feats=n_feat, hid_feats=n_hidden, out_feats=n_class, nums_layers=n_layer,
                  activation=F.relu, prob_dropout=dropout)
# optimizer
optimizer = torch.optim.Adam(params=model.parameters(), lr=lr, weight_decay=weight_decay)

# train model
for epoch in range(n_epochs):
    model.train()
    # calculate loss and accuracy
    logits = model(feats)
    loss = F.cross_entropy(logits[train_mask], labels[train_mask])
    acc = evaluate(model, feats, labels, valid_mask)
    # back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # print
    if epoch % 10 == 0:
        print('Epoch {:3d} | Loss {:.4f} | Accuracy {:.4f}'. format(epoch, loss.item(), acc))

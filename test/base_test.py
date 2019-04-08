import numpy
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

import dgl_data_parallel as dglp

numpy.set_printoptions(threshold='nan')

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

###############################################################################
# We then define the node UDF for ``apply_nodes``, which is a fully-connected layer:

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

###############################################################################
# We then proceed to define the GCN module. A GCN layer essentially performs
# message passing on all the nodes then applies the `NodeApplyModule`. Note
# that we omitted the dropout in the paper for simplicity.

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

###############################################################################
# The forward function is essentially the same as any other commonly seen NNs
# model in PyTorch.  We can initialize GCN like any ``nn.Module``. For example,
# let's define a simple neural network consisting of two GCN layers. Suppose we
# are training the classifier for the cora dataset (the input feature size is
# 1433 and the number of classes is 7).

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, F.relu)

    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
net = Net()
print(net)

###############################################################################
# We load the cora dataset using DGL's built-in data module.

from dgl.data import citation_graph as citegrh
def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    mask = th.ByteTensor(data.train_mask)
    g = DGLGraph(data.graph)
    print("g: ")
    print(g.all_edges())
    print(g.edata)
    print("features: ")
    print(features.size())
    print(features)
    print("labels: ")
    print(labels.size())
    print(labels)
    print("mask: ")
    print(mask.size())
    print(mask)
    return g, features, labels, mask

###############################################################################
# We then train the network as follows:

import time
import numpy as np
g, features, labels, mask = load_cora_data()
device_ids = [0,1]
dur = []
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
net = dglp.DataParallel(net)
#net = net.cuda()
#output = net(g,features.cuda())
features = features.cuda()
labels = labels.cuda()
net = net.cuda()

print(labels[mask].size())
print(labels[mask])

for epoch in range(30):
    if epoch >=3:
        t0 = time.time()

    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >=3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), np.mean(dur)))

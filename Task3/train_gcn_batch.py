

from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import os

import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

from pygcn.utils import load_data, accuracy, setup_seed, load_data_house
from pygcn.models import GCN

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

setup_seed(args.seed)


# Load data
adj, features, labels, idx_train, idx_val, idx_test, min_max_scalers = load_data_house()

print(adj.shape, features.shape)
# assert False

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            # nclass=labels.max().item() + 1,
            nclass=1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


train_data = torch.cat([features[idx_train], labels[idx_train]], dim=-1)
dataloaer = DataLoader(train_data, batch_size=256)
val_features, val_labels = features[idx_val].cuda(), labels[idx_val].cuda()
train_features, train_labels = features[idx_train].cuda(), labels[idx_train].cuda()

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    # idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()

    for i, data in enumerate(dataloaer):
      batch_features, batch_labels = data[:, :-1].cuda(), data[:, -1:].cuda()

      model.train()
      optimizer.zero_grad()
      output = model(batch_features, None)
      

      # loss_train = F.nll_loss(output[idx_train], labels[idx_train])
      # print(output.shape, labels.shape, idx_train.shape)
      loss_train = F.mse_loss(output, batch_labels)
      # acc_train = accuracy(output[idx_train], labels[idx_train])
      loss_train.backward()
      optimizer.step()

      if not args.fastmode:
          # Evaluate validation set performance separately,
          # deactivates dropout during validation run.
          model.eval()
          output = model(batch_features, None)

      # loss_val = F.nll_loss(output[idx_val], labels[idx_val])
      # loss_val = F.mse_loss(output[idx_val], labels[idx_val])
      # acc_val = accuracy(output[idx_val], labels[idx_val])

      # loss_train = loss_train.item()
      # loss_val = loss_val.item()

      output_np = min_max_scalers[-1].inverse_transform(output.data.cpu().numpy())
      labels_np = min_max_scalers[-1].inverse_transform(batch_labels.data.cpu().numpy())
      # loss_val = np.sqrt(mean_squared_error(output_np[idx_val.cpu().numpy()], labels_np[idx_val.cpu().numpy()]))
      loss_train = np.sqrt(mean_squared_error(output_np, labels_np))
    

    model.eval()
    all_features = torch.cat([val_features, train_features], axis=0)
    all_output = model(all_features, None)
    val_output = all_output[:len(val_features), :]
    output_np = min_max_scalers[-1].inverse_transform(val_output.data.cpu().numpy())
    labels_np = min_max_scalers[-1].inverse_transform(val_labels.data.cpu().numpy())
    loss_val = np.sqrt(mean_squared_error(output_np, labels_np))
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
        #   'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
        #   'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
# test()

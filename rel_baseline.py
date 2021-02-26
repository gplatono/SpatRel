import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader
import random
import numpy as np
import sys

num_classes = 16

class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


def build_data_loader(batch_size, num_workers, is_shuffle, test_perc):
    data = None
    with open('dataset', 'r') as file:
        data = file.readlines()
    data = [json.loads(item) for item in data ]
    data = [item for item in data if (abs(item['label']) == 3) or (abs(item['label']) == 3)]
    random.shuffle(data)
    samples = []
    labels = []
    test_s = []
    test_l = []
    label_count = [0]*num_classes
    for item in data:
        label_count[abs(item['label'])-1] += 1
    for i in range(len(label_count)):
        label_count[i] *= 1-test_perc
    print(label_count)
    for i, item in enumerate(data):
        try:
            if item['label'] > 0:
                label = 1
            elif item['label'] < 0:
                label = 0
            else:
                print("wrong label:", item['label'])
                exit()
        except:
            print(item['label'] - 1)
            exit()
        sample = []
        for arg in item['arg0']:
            sample += arg[2:3]
        for arg in item['arg1']:
            sample += arg[2:3]

        if label_count[abs(item['label'])-1] > 0:
            labels.append(torch.tensor([label], dtype=torch.float32))
            samples.append(torch.tensor(sample, dtype=torch.float32))
            label_count[abs(item['label'])-1] += -1
        else:
            test_l.append(torch.tensor([label], dtype=torch.float32))
            test_s.append(torch.tensor(sample, dtype=torch.float32))
    #print(label_count)
    #print('labels: ', labels)
    #print('label count: ', len(labels))
    dataset = Dataset(samples, labels)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
    test_d = Dataset(test_s, test_l)
    test_loader = DataLoader(test_d, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
    return train_loader, test_loader


class Net1(nn.Module):
    def __init__(self, n_in, n_out, layers):
        super(Net1, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.mlp = []
        for i in range(len(layers)+1):
            if i == 0:
                self.mlp += [nn.Linear(n_in, layers[i])]
            elif i == len(layers):
                self.mlp += [nn.Linear(layers[i-1], n_out)]
            else:
                self.mlp += [nn.Linear(layers[i-1], layers[i])]
            self.mlp += [nn.ReLU(inplace=False)]
        self.mlp = nn.ModuleList(self.mlp)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        for i in range(len(self.mlp)):
            z = self.mlp[i](z)
        z = self.sigmoid(z)
        return z

def train(train_loader, net, optimizer, criterion, epoch):
    for epoch_i in range(epoch):
        epoch_loss = 0.0
        count = 0
        correct = 0
        fcorrect = 0
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            for j in range(len(outputs)):
                if abs(outputs[j]-labels[j]) < 0.5:
                    correct += 1
                count += 1
            for j in range(len(outputs)):
                if inputs[j][0] < inputs[j][1]:
                    if labels[j] == 1:
                        fcorrect += 1
                else:
                    if labels[j] == 0:
                        fcorrect += 1

            loss.backward()
            optimizer.step()
            epoch_loss += loss

        # suppress divide by 0 message

        t = np.array(correct) / np.array(count)
        print("acc: ", np.sum(np.array(correct))/np.sum(np.array(count)))
        print("facc: ", fcorrect/count)
        print('epoch %d loss: %.3f' % (epoch_i+1, epoch_loss / len(train_loader)))


def test(test_loader, net):
    count = [0]*num_classes
    correct = [0]*num_classes
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            outputs = net(inputs)
            for j in range(len(outputs)):
                if abs(outputs[j] - labels[j]) < 0.5:
                    correct += 1
                count += 1

    t = np.array(correct) / np.array(count)
    print("testing")
    print("acc: ", np.sum(np.array(correct)) / np.sum(np.array(count)))
    return t.tolist(), (np.sum(np.array(correct)) / np.sum(np.array(count))).tolist()

def run1(layers):
    # build dataloader
    train_loader, test_loader = build_data_loader(batch_size=4, num_workers=2, is_shuffle=True, test_perc=0.1)
    # initialize model
    # net = Net(7, 1)
    net = Net1(2, 1, layers)
    # use crossentropy loss
    #criterion = nn.MSELoss()
    criterion = nn.BCELoss()
    # use sgd optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
    # train
    train(train_loader, net, optimizer, criterion, epoch=20)
    # save model
    savepath = 'classifier_param.pth'
    torch.save(net.state_dict(), savepath)
    # test
    return test(test_loader, net)

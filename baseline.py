# import torch
# import torch.nn as nn
# import torch.optim as optim
# import json
# from torch.utils.data import DataLoader

# num_classes = 15
# in_dim = 6

# class Dataset(torch.utils.data.Dataset):
#   def __init__(self, samples, labels):
#         self.labels = labels
#         self.samples = samples

#   def __len__(self):
#         return len(self.samples)

#   def __getitem__(self, index):
#     #print ('item: ', self.samples[index], self.labels[index])
#     return self.samples[index], self.labels[index]


# def build_data_loader(batch_size, num_workers, is_shuffle):
#     data = None
#     with open('dataset', 'r') as file:
#         data = file.readlines()
#     data = [json.loads(item) for item in data]
#     samples = []
#     labels = []
#     for item in data:
#         label = [0] * num_classes
#         if item['label'] != 12 and item['label'] != -12:
#             if item['label'] > 0:
#                 label[item['label']-1] = 1
#             else:
#                 label[item['label']-1] = 0
#             labels.append(torch.tensor(label, dtype=torch.float32))
#             sample = []
#             for arg in item['arg0']:
#                 sample += arg[:3]
#             for arg in item['arg1']:
#                 sample += arg[:3]
#             #print('samples: ', sample)
#             samples.append(torch.tensor(sample, dtype=torch.float32))
#         else:
#             continue

#     #print('labels: ', labels)
#     #print('label count: ', len(labels))
#     dataset = Dataset(samples, labels)
#     train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
#     test_loader = train_loader

#     return train_loader, test_loader


# class Net(nn.Module):
#     def __init__(self, n_in, n_out):
#         super(Net, self).__init__()
#         self.n_in = n_in
#         self.n_out = n_out
#         self.linear1 = nn.Linear(n_in, 2)
#         self.hidden_1 = nn.Linear(2, 3)
#         self.hidden_2 = nn.Linear(3, n_out)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, z):
#         z = self.linear1(z)
#         z = self.hidden_1(z)
#         z = self.hidden_2(z)
#         z = self.sigmoid(z)
#         return z


# def train(train_loader, net, optimizer, criterion, epoch):
#     epoch_loss = 0.0
#     for epoch_i in range(epoch):
#         for i, data in enumerate(train_loader):
#             inputs, labels = data
#             print ("inp: ", inputs, "label: ", labels)
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += outputs.shape[0] * loss.item()
#         # print loss
#         print('epoch %d loss: %.3f' % (epoch+1, epoch_loss / len(train_loader)))


# def test(test_loader, net):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for data in enumerate(test_loader):
#             inputs, labels = data
#             outputs = net(inputs)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     # print accuracy
#     print('Accuracy of the network: %.3f' % (correct/total))



# if __name__ == '__main__':
#     # build dataloader
#     train_loader, test_loader = build_data_loader(batch_size=4, num_workers=2, is_shuffle=True)
#     # initialize model
#     net = Net(n_in = in_dim, n_out = num_classes)
#     # use crossentropy loss
#     criterion = nn.BCELoss()
#     # use sgd optimizer
#     optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#     # train
#     train(train_loader, net, optimizer, criterion, epoch=6)
#     # save model
#     savepath = 'classifier_param.pth'
#     torch.save(net.state_dict(), savepath)

import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader
import random
import numpy as np
import sys

num_classes = 16
int_to_anno = {}

class Dataset(torch.utils.data.Dataset):
    def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]


# class DatasetIterable:
#     def __init__(self, Dataset):
#         self._Dataset = Dataset
#         self._index = 0
#
#     def __next__(self):
#         if self._index >= len(self._Dataset):
#             raise StopIteration
#         data = self._Dataset.samples[self._index], self._Dataset.labels[self._index]
#         self._index += 1
#         return data


def build_data_loader(batch_size, num_workers, is_shuffle, test_perc):
    data = None
    with open('dataset', 'r') as file:
        data = file.readlines()
    data = [json.loads(item) for item in data ]
    data = [item for item in data if (abs(item['label']) == 13) or (abs(item['label']) == 13)]
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
    #print(label_count)
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
            sample += arg[:11]
        for arg in item['arg1']:
            sample += arg[:11]
        if item['label'] != 12 and item['label'] != -12:
            sample += [0]*11

        label_encode = [0]*num_classes
        label_encode[abs(item['label'])-1] += 1
        sample += label_encode
        int_to_anno[i] = item['annotation']
        #sample += [abs(item['label'])]
        #print('samples: ', sample)
        #print('labels: ', label)
        # if i < len(data) * (1-test_perc):
        #     labels.append(torch.tensor([label], dtype=torch.float32))
        #     samples.append(torch.tensor(sample, dtype=torch.float32))
        # else:
        #     test_l.append(torch.tensor([label], dtype=torch.float32))
        #     test_s.append(torch.tensor(sample, dtype=torch.float32))

        if label_count[abs(item['label'])-1] > 0:
            labels.append(torch.tensor([label], dtype=torch.float32))
            samples.append(torch.tensor(sample, dtype=torch.float32))
            label_count[abs(item['label'])-1] += -1
        else:
            test_l.append(torch.tensor([label]+[i], dtype=torch.float32))
            test_s.append(torch.tensor(sample, dtype=torch.float32)) #add annotation index
    #print(label_count)
    #print('labels: ', labels)
    #print('label count: ', len(labels))
    dataset = Dataset(samples, labels)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
    test_d = Dataset(test_s, test_l)
    test_loader = DataLoader(test_d, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
    return train_loader, test_loader


class Net(nn.Module):
    def __init__(self, n_in, n_out):
        super(Net, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.hidden_0 = nn.Linear(n_in, 5)
        self.relu = nn.ReLU()
        self.hidden_1 = nn.Linear(5, 7)
        self.hidden_2 = nn.Linear(7, 7)
        #self.hidden_3 = nn.Linear(11, 7)
        self.hidden_4 = nn.Linear(7, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.hidden_0(z)
        #z = self.relu(z)
        z = self.hidden_1(z)
        #z = self.relu(z)
        z = self.hidden_2(z)
        #z = self.relu(z)
        #z = self.hidden_3(z)
        #z = self.relu(z)
        z = self.hidden_4(z)
        #z = self.relu(z)
        z = self.sigmoid(z)
        return z

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
        count = [0]*num_classes
        correct = [0]*num_classes
        for i, data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            for j in range(len(outputs)):
                label_index=(inputs[j][-num_classes:]==1).nonzero(as_tuple=True)[0]
                #print(inputs[j])
                #print(label_index)
                if abs(outputs[j]-labels[j]) < 0.5:
                    correct[label_index] += 1
                count[label_index] += 1
            loss.backward()
            optimizer.step()
            epoch_loss += loss

        # suppress divide by 0 message
        fake_count = []
        for i in range(len(count)):
            if count[i] == 0:
                fake_count += [1]
            else:
                fake_count += [count[i]]
        t = np.array(correct) / np.array(fake_count)
        print("relation acc: ", t.tolist())
        print("acc: ", np.sum(np.array(correct))/np.sum(np.array(count)))
        # print loss
        print('epoch %d loss: %.3f' % (epoch_i+1, epoch_loss / len(train_loader)))


def test(test_loader, net, test):
    count = [0]*num_classes
    correct = [0]*num_classes
    wrong_anno = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            anno = labels[:,1:]
            labels = labels[:,:1]
            outputs = net(inputs)
            for j in range(len(outputs)):
                label_index = (inputs[j][-num_classes:] == 1).nonzero(as_tuple=True)[0]
                if abs(outputs[j] - labels[j]) < 0.5:
                    correct[label_index] += 1 #label correctly predicted
                else:
                    wrong_anno += [int(anno[j])]
                count[label_index] += 1

    #suppress divide by 0 message
    fake_count = []
    for i in range(len(count)):
        if count[i]==0:
            fake_count+=[1]
        else:
            fake_count+=[count[i]]
    t = np.array(correct) / np.array(fake_count)
    print("testing")
    print("relation acc: ", t.tolist())
    print("acc: ", np.sum(np.array(correct)) / np.sum(np.array(count)))
    if test:
        for index in wrong_anno:
            print(int_to_anno[index])
        input()
    return t.tolist(), (np.sum(np.array(correct)) / np.sum(np.array(count))).tolist()


def run(layers):
    # build dataloader
    train_loader, test_loader = build_data_loader(batch_size=4, num_workers=2, is_shuffle=True, test_perc=0.1)
    # initialize model
    # net = Net(7, 1)
    net = Net1(33+num_classes, 1, layers)
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
    return test(test_loader, net, True)

if __name__ == '__main__':
    # build dataloader
    train_loader, test_loader = build_data_loader(batch_size=4, num_workers=2, is_shuffle=True, test_perc=0.1)
    # initialize model
    net = Net(33+num_classes, 1)
    #net = Net1(10, 1)
    # use crossentropy loss
    criterion = nn.BCELoss()
    # use sgd optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9)
    # train
    train(train_loader, net, optimizer, criterion, epoch=10)
    # save model
    savepath = 'classifier_param.pth'
    torch.save(net.state_dict(), savepath)
    # test
    test(test_loader, net, True)

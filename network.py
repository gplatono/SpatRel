import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json


num_classes = 15

class Dataset(torch.utils.data.Dataset):
  def __init__(self, samples, labels):
        self.labels = labels
        self.samples = samples

  def __len__(self):
        return len(self.samples)

  def __getitem__(self, index):
    #print ('item: ', self.samples[index], self.labels[index])
    return self.samples[index], self.labels[index]


def build_data_loader1(batch_size, num_workers, is_shuffle, relation):

    rel_to_label = {'on': 14, 'to the left of': 1, 'left of': 1, 'to the right of': 2, 'right of': 2, 'above': 3,
            'below': 4, 'in front of': 5, 'behind': 6, 'over': 7, 'under': 8, 'underneath': 8, 'in': 9, 'inside': 9,
            'touching': 10, 'touch': 10, 'at': 11, 'next to': 11, 'between': 12, 'near': 13, 'on top of': 14, 'beside': 15, 'besides': 15}
    relation = rel_to_label[relation]

    data = None
    with open('dataset', 'r') as file:
        data = file.readlines()
    data = [json.loads(item) for item in data]
    samples = []
    labels = []
    for item in data:
        if item['label'] == relation:
            label = [1]
        elif item['label'] == -relation:
            label = [0]
        else:
            continue
        labels.append(torch.tensor(label, dtype=torch.float32))
        sample = []
        for arg in item['arg0']:
            sample += arg[:3]
        for arg in item['arg1']:
            sample += arg[:3]
        #sample += item['observer_loc']
        #sample += item['observer_front']
        print('samples: ', sample)
        samples.append(torch.tensor(sample, dtype=torch.float32))
    print('labels: ', labels)
    print('label count: ', len(labels))
    #input()
    dataset = Dataset(samples, labels)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
    test_loader = train_loader
    return train_loader, test_loader


# data loader for multi-class
def build_data_loader2(batch_size, num_workers, is_shuffle):
    data = None
    with open('dataset', 'r') as file:
        data = file.readlines()
    data = [json.loads(item) for item in data]
    samples = []
    labels = []
    for item in data:
        label = [0] * num_classes
        if item['label'] > 0 and item['label'] != 12:
            label[item['label']-1] = 1
            labels.append(torch.tensor(label, dtype=torch.float32))
            sample = []
            for arg in item['arg0']:
                sample += arg[:3]
            for arg in item['arg1']:
                sample += arg[:3]
            print('samples: ', sample)
            samples.append(torch.tensor(sample, dtype=torch.float32))
        else:
            continue

    print('labels: ', labels)
    print('label count: ', len(labels))
    dataset = Dataset(samples, labels)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
    test_loader = train_loader
    return train_loader, test_loader

# data loader for multi-label
def build_data_loader3(batch_size, num_workers, is_shuffle):
    data = None
    with open('dataset', 'r') as file:
        data = file.readlines()
    data = [json.loads(item) for item in data]
    samples = []
    labels = []
    for item in data:
        label = [0] * num_classes
        if item['label'] != 12 and item['label'] != -12:
            if item['label'] > 0:
                label[item['label']-1] = 1
            else:
                label[item['label']-1] = 0
            labels.append(torch.tensor(label, dtype=torch.float32))
            sample = []
            for arg in item['arg0']:
                sample += arg[:3]
            for arg in item['arg1']:
                sample += arg[:3]
            print('samples: ', sample)
            samples.append(torch.tensor(sample, dtype=torch.float32))
        else:
            continue

    print('labels: ', labels)
    print('label count: ', len(labels))
    dataset = Dataset(samples, labels)
    train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
    test_loader = train_loader
    return train_loader, test_loader


class Net1(nn.Module):
    def __init__(self, n_in, n_out):
        super(Net1, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.hidden1 = nn.Linear(n_in, 10)
        self.hidden2 = nn.Linear(10, 5)
        self.output = nn.Linear(5, n_out)
        self.sigmoid = nn.Sigmoid()
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, z):
        z = self.hidden1(z)
        z = self.hidden2(z)
        z = self.output(z)
        z = self.sigmoid(z)
        return z

    
# generalized version of base-line model
class Net2(nn.Module):
    def __init__(self, n_in, n_out, h_shapes=[]):
        super(Net2, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.input_layer = nn.Linear(n_in, h_shapes[0])
        self.hidden_layers = nn.ModuleList()
        for k in range(0, len(h_shapes) - 1):
            self.hidden_layers.append(nn.Linear(h_shapes[k], h_shapes[k + 1]))
        self.output_layer = nn.Linear(h_shapes[len(h_shapes)-1], n_out)

    def forward(self, z):
        z = self.input_layer(z)
        for layer in self.hidden_layers:
            z = layer(z)
        z = self.output_layer(z)
        #z = nn.softmax(self.output_layer(z), dim=1)
        return z

    
def train(train_loader, net, optimizer, criterion, epoch):
    print("start training")
    epoch_loss = 0.0
    for epoch_i in range(epoch):
        # print ("INIT")
        # iterator = iter(train_loader)
        # x_b, y_b = iterator.next()
        # print (x_b, y_b)
        epoch_loss = 0
        epoch_acc = 0

        for step, data in enumerate(train_loader):
            inputs, labels = data
            print (inputs, labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            # labels = torch.max(labels, 1)[1]  # transfer label to 1D target when using CrossEntropy
            print('outputs: ', outputs)
            print('labels: ', labels)
            rnd = torch.round(outputs)
            print('rounded outputs: ', rnd)
            acc = torch.round(torch.abs(outputs - labels))
            acc = 1 - torch.sum(acc) / torch.numel(acc)
            print('batch acc: ', acc * 100)
            epoch_acc += acc
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # print loss
        print('epoch %d acc: %.3f' % (epoch_i+1, 100 * epoch_acc / len(train_loader)))
        print('epoch %d loss: %.3f' % (epoch_i+1, epoch_loss / len(train_loader)))
        #input()


def test(test_loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for step, data in enumerate(test_loader):
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print accuracy
    print('Accuracy of the network: %.3f' % (correct/total))

if __name__ == '__main__':
    # build dataloader
    # train_loader, test_loader = build_data_loader(batch_size=10, num_workers=1, is_shuffle=True, relation='near')
    """See rel_to_label dictionary for the set of options for 'relation' """
    train_loader, test_loader = build_data_loader3(batch_size=10, num_workers=1, is_shuffle=True)
    # initialize model
    
    net = Net1(n_in=6, n_out=15)
    # net = Net2(n_in=72, n_out=1, h_shapes=[50, 40])

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.BCELoss()
    # criterion = nn.MSELoss()
    # use sgd optimizer
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # train
    train(train_loader, net, optimizer, criterion, epoch=100)
    # save model
    savepath = 'classifier_param.pth'
    torch.save(net.state_dict(), savepath)


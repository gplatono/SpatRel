import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import DataLoader

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


def build_data_loader(batch_size, num_workers, is_shuffle):
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


class Net(nn.Module):
    def __init__(self, n_in, n_out):
        super(Net, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear1 = nn.Linear(n_in, 2)
        self.hidden_1 = nn.Linear(2, 3)
        self.hidden_2 = nn.Linear(3, n_out)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.linear1(z)
        z = self.hidden_1(z)
        z = self.hidden_2(z)
        z = self.sigmoid(z)
        return z


def train(train_loader, net, optimizer, criterion, epoch):
    epoch_loss = 0.0
    for epoch_i in range(epoch):
        for data in enumerate(train_loader):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += outputs.shape[0] * loss.item()
        # print loss
        print('epoch %d loss: %.3f' % (epoch+1, epoch_loss / len(train_loader)))


def test(test_loader, net):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in enumerate(test_loader):
            inputs, labels = data
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # print accuracy
    print('Accuracy of the network: %.3f' % (correct/total))



if __name__ == '__main__':
    # build dataloader
    train_loader = build_data_loader(batch_size=4, num_workers=2, is_shuffle=True)
    # initialize model
    net = Net()
    # use crossentropy loss
    criterion = nn.CrossEntropyLoss()
    # use sgd optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # train
    train(train_loader, net, optimizer, criterion, epoch=6)
    # save model
    savepath = 'classifier_param.pth'
    torch.save(net.state_dict(), savepath)


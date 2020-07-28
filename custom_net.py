import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json

num_classes = 15


class Sample:
    def __init__(self, val):
        self.val = val
        self.centroid = (val, val, val)
        self.bbox = 0

#For custom objects
# def build_custom_loader(batch_size, num_workers, is_shuffle):
#     samples = []
#     labels = []
#     for i in range(1000):
#         samples.append(Sample(i))
#         labels.append(i * i)
#     dataset = Dataset(samples, labels)
#     train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = is_shuffle, num_workers = num_workers)
#     test_loader = train_loader
#     return train_loader, test_loader

class CustomNet:

    def __init__(self):
        self.params = torch.tensor([1, 1, 1], dtype=torch.float32, requires_grad = True)

    def compute(self, sample):
        result = torch.dot(torch.tensor(sample.centroid, dtype=torch.float32), self.params)
        return result

    def parameters(self):
        return [self.params]

if __name__ == '__main__':


    net = CustomNet()

    train_size = 1000
    batch_size = 20
    epochs = 10000

    samples = []
    labels = []
    for i in range(train_size):
        samples.append(Sample(i))
        labels.append(torch.tensor(i, dtype = torch.float32))

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    idx = 0
    for epoch in range(epochs):
        batch_loss = 0
        for j in range(batch_size):            

            optimizer.zero_grad()
            outputs = net.compute(samples[idx])            
            batch_loss += torch.sum(torch.square(outputs - labels[idx]))
            idx = idx + 1 if idx < train_size - 1 else 0
            
        batch_loss /= batch_size
        batch_loss.backward()
        optimizer.step()
        print ("Epoch: %d, loss: %.3f" % (epoch, batch_loss))
        print ("Params: ", net.params)
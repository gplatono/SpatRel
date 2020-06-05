import torch
import torch.nn as nn
import torch.optim as optim

def build_data_loader(batch_size, num_workers, is_shuffle):
    # TODO
    return train_loader, test_loader


class Net1(nn.Module):
    def __init__(self, n_in, n_out):
        super(Net1, self).__init__()
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

    
# generalized version
class Net2(nn.Module):
    def __init__(self, n_in, n_out, h_shapes=[]):
        super(Net2, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.input_layer = nn.Linear(n_in, h_shapes[0])
        self.hidden_layers = nn.ModuleList()
        for k in range(1, len(h_shapes) - 1):
            self.hidden_layers.append(nn.Linear(h_shapes[k], h_shapes[k + 1]))
        self.output_layer = nn.Linear(h_shapes[len(h_shapes)], n_out)

    def forward(self, z):
        z = self.input_layer(z)
        for layer in self.hidden_layers:
            z = layer(z)
        z = nn.softmax(self.output_layer(z), dim=1)
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
    net = Net1()
    # use crossentropy loss
    criterion = nn.CrossEntropyLoss()
    # use sgd optimizer
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    # train
    train(train_loader, net, optimizer, criterion, epoch=6)
    # save model
    savepath = 'classifier_param.pth'
    torch.save(net.state_dict(), savepath)


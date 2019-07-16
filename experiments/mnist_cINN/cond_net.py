import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import config as c
import data as color_data

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
                     nn.Conv2d(1, 32, kernel_size=3),
                     nn.Conv2d(32, 64, kernel_size=3),
                     nn.MaxPool2d(2),
                     nn.Conv2d(64, 64, kernel_size=3),
                     nn.Conv2d(64, 64, kernel_size=3),
                     nn.MaxPool2d(2),
                     )

        self.linear = nn.Sequential(
                    nn.Dropout(),
                    nn.Linear(1024, 512),
                    nn.Dropout(),
                    nn.Linear(512, 512),
                    nn.Dropout(),
                    nn.Linear(512, c.cond_width),
                    )

        self.fc_final = nn.Linear(c.cond_width, 10)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(c.batch_size, -1)
        x = self.linear(x)
        x = self.fc_final(x)
        return F.log_softmax(x, dim=1)

    def features(self, x):
        x = self.conv(x)
        x = x.view(c.batch_size, -1)
        return self.linear(x)

model = Net().cuda()
log_interval = 25

def train():
    model.train()
    for batch_idx, (color, target, data) in enumerate(color_data.train_loader):
        data, target = data.cuda(), target.long().cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(color_data.train_loader.dataset),
                100. * batch_idx / len(color_data.train_loader), loss.item()))

test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('./mnist_data', train=False, transform=transforms.ToTensor()),
        batch_size=c.batch_size, shuffle=True, drop_last=True)

def test():
    model.train()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


if __name__ == '__main__':

    optimizer = optim.SGD(model.parameters(), lr=0.03, momentum=0.5)

    for epoch in range(6):
        train()
        test()

    torch.save(model.state_dict(), c.cond_net_file)

else:
    model.train()
    if c.cond_net_file:
        model.load_state_dict(torch.load(c.cond_net_file))

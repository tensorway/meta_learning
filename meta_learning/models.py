import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, nways):
        super().__init__()
        self.pixs = 27**2
        self.conv1 = nn.Conv2d(1, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.conv4 = nn.Conv2d(64,128, 3)
        self.pad   = nn.ZeroPad2d(1)

        self.pool = nn.MaxPool2d(2)
        self.lin1 = nn.Linear(512*4, 32)
        self.lin2 = nn.Linear(32, nways)

    def forward(self, x):
        x0 = x.shape
        x = x.view(-1, 1, x.shape[-2], x.shape[-1])
        h = self.conv1(x)
        h = self.pool(h)
        h = F.relu(h)

        h = self.conv2(h)
        h = self.pool(h)
        h = F.relu(h)

        h2 = self.pad(h)
        h2 = self.conv3(h2)
        h2 = F.relu(h2)
        h = h2

        h2 = self.pad(h)
        h2 = self.conv4(h2)
        h2 = F.relu(h2)
        h = h2

        h = h.view(h.shape[0], -1)
        h = self.lin1(h)
        h = F.relu(h)
        h = self.lin2(h)
        h = torch.softmax(h, dim=-1)
        return h.view(x0[0], x0[1], h.shape[-1])

class SimpleMLP(nn.Module):
    def __init__(self, net_arch):
        super().__init__()
        l = [nn.Linear(a, b) for a, b in zip(net_arch[:-1], net_arch[1:])]
        self.layers = nn.ModuleList(l)
    def forward(self, x):
        h = torch.tensor(x, dtype=torch.float)
        for lay in self.layers[:-1]:
            h = F.tanh(lay(h))
        h = self.layers[-1](h)
        return h



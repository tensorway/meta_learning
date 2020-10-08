# %%
import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
nways = 2

455
# %%
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, k, n, ntrain, is_train):
        super().__init__()
        self.n_examples_per_class = 20
        self.dataset = torchvision.datasets.Omniglot(
            root="./data", download=True, transform=torchvision.transforms.ToTensor()
        )
        self.nclasses = len(self.dataset)//self.n_examples_per_class
        self.ntrain = ntrain
        self.is_train = is_train
        self.n = n
        self.k = k
        assert ntrain <= self.nclasses, "ntrain examples should be less than or equal to total number of classes(" + str(self.nclasses)+")"
    def __len__(self):
        return 1234
    def get_one_random(self, iclass):
        i2 = random.randint(0, self.n_examples_per_class-1)
        a, _ = self.dataset[iclass*self.n_examples_per_class + i2]
        return a[:, ::4, ::4].unsqueeze(0).unsqueeze(0) #, torch.tensor([[b]]).unsqueeze(0).unsqueeze(0)
            
    def __getitem__(self, idx):
        llt1=[]; rng=0
        if self.is_train:
            rlist = random.sample(range(0, self.ntrain-1), self.n)
        else:
            rlist = random.sample(range(self.ntrain, self.nclasses), self.n)
        for nk in range(self.k):
            lt1=[]; lt2=[]
            for iclass in rlist:
                t1 = self.get_one_random(iclass)
                lt1.append(t1)
            llt1.append(torch.cat(lt1, dim=1))
        t1 = torch.cat(llt1, dim=0).squeeze(-3)
        
        l = list(range(0, self.n))
        t2 = torch.tensor([l])
        t2 = t2.repeat(self.k, 1, 1)
        t2 = F.one_hot(t2, num_classes=nways).squeeze(-3)
            
        return t1, t2
        

bsize = 512
ksize = 1
dataset = MyDataset(k=ksize+1, n=nways, ntrain=800, is_train=True)
dataset_test = MyDataset(k=ksize, n=nways, ntrain=800, is_train=False)

loader = torch.utils.data.DataLoader(dataset, batch_size=bsize, num_workers=3)
      
# image, label = dataset[50]
# print(label)  # int1
# plt.imshow(image.numpy()[0])
for t in loader:
    print(t[0].shape, t[1].shape)
    ns = t[0][0, 0, 0, :, :].numpy()
    # print(model(t[0], t[1]).shape)
    plt.imshow(ns)
    break# next(loader)





# %%
class Model(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.pixs = 27**2
        self.lin1 = nn.Linear(512+self.pixs, 64)
        self.lin2 = nn.Linear(64, nways)

        self.lstm = nn.LSTM(self.pixs+nways, 512, num_layers=2, batch_first=True)
    def forward(self, x, idx, x2):
        # print(x.shape, idx.shape, x2.shape)
        # print(idx)
        #num_layers, num_directions, batch, hidden_size)

        x = x.view(x.shape[0], x.shape[1], x.shape[2], self.pixs).view(x.shape[0], -1, self.pixs)
        idx = idx.view(idx.shape[0], -1, nways).float()
        # print(idx[:2])
        x2 = x2.view(x2.shape[0], nways, self.pixs)
        # print(x.shape, idx.shape, x2.shape)

        h = torch.cat((x, idx), dim=-1)
        _, (h, _) = self.lstm(h)
        # print(h.shape)
        h = h.view(2, 1, x.shape[0], -1)[1].squeeze(0).unsqueeze(1).repeat(1, nways, 1)
        # print(h.shape, x2.shape)
        h2 = torch.cat((x2, h), dim=-1)
        h2 = self.lin1(h2)
        h2 = F.relu(h2)
        h2 = self.lin2(h2)
        h2 = torch.softmax(h2, dim=-1)
        return h2
        
model = Model()



# %%
opt = torch.optim.Adam(model.parameters(), lr=1e-2)


# %%
ename = 'tst_hdim512_b512_long'
os.system('mkdir tb/'+ename)
writer = SummaryWriter('tb/'+ename)

ii = 0
eps=1000
for ep in range(eps):
    for step, (x, y) in enumerate(loader):
        perm = torch.randint(0, nways, (nways,))
        # x = x[:, :, perm];   y= y[:, :, perm]
        # print(x.shape); 
        xexamples, xtoguess = x[:, :-1], x[:, -1]
        yexamples, ytoguess = y[:, :-1], y[:, -1]
        # print(xexamples.shape, yexamples.shape);
        preds = model(xexamples, yexamples, xtoguess)
        # print(preds[:2])
        loss = -(ytoguess*torch.log(preds+1e-8)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        accu = acc(ytoguess, preds)
        writer.add_scalar("loss/accuracy", accu, ii:=ii+1)
        writer.add_scalar("loss/loss", loss.item(), ii)
        if step%4==0:
            print(ep, step, loss.item(), accu)
# %%

def acc (goal, guess):
    acc = (torch.argmax(goal, dim=-1)==torch.argmax(guess, dim=-1)).float().mean()
    # print("goal", goal)
    # print("guess", guess>0.5)
    return acc.item()

# %%
a = torch.randn(10, 10)
idx = torch.randint(0, 10, (10,))
a[:, idx], a, idx
# %%
a, b = dataset[0]
a.shape
# %%
plt.imshow(a[1, 0].numpy())

# %%

# %%


# %%

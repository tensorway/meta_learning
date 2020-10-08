import torchvision
import torch 
import torch.nn as nn
import torch.nn.functional as F
import random

class OmniglotDataset(torch.utils.data.Dataset):
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
        t2 = F.one_hot(t2, num_classes=self.n).squeeze(-3)
            
        return t1, t2

def acc (goal, guess):
    acc = (torch.argmax(goal, dim=-1)==torch.argmax(guess, dim=-1)).float().mean()
    return acc.item()


class SinusoidalDataset(torch.utils.data.Dataset):
    def __init__(self, bsize_outer, bsize_inner, minA=1, maxA=10, minP=3, maxP=4*3.14, minO=-2, maxO=2):
        super().__init__()
        self.bsize_outer = bsize_outer
        self.bsize_inner = bsize_inner
        self.minA, self.maxA = minA, maxA
        self.minP, self.maxP = minP, maxP
        self.minO, self.maxO = minO, maxO

    def __len__(self):
        return 1234

    def __getitem__(self, idx):
        amplitudes = torch.rand(self.bsize_outer)*(self.minA-self.maxA) - self.minA
        periods    = torch.rand(self.bsize_outer)*(self.minP-self.maxP) - self.minP
        offsets    = torch.rand(self.bsize_outer)*(self.minO-self.maxO) - self.minO
        x, y = [], []
        for a, p, o in zip(amplitudes, periods, offsets):
            tmpx, tmpy = Sinusoid(a, p, o).get_sample(self.bsize_inner) 
            x.append(tmpx)
            y.append(tmpy)
        return torch.cat(x, dim=0), torch.cat(y, dim=0)

class Sinusoid:
    def __init__(self, amplitude, period, offset):
        self.amplitude = amplitude
        self.period = period
        self.offset = offset
    def get_sample(self, k=1, noise=0, minx=-10, maxx=10, x=None):
        if x is None:
            x = torch.rand(1, k, 1)*(maxx-minx)+minx
        y = self.amplitude*torch.sin(x/self.period*2*3.14 + self.offset)
        y += torch.randn_like(y)*noise
        return x, y

def regression_acc (goal, guess):
    acc = ((goal-guess)**2).mean()
    return acc.item()
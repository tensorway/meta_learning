import torch

def acc_classification (goal, guess):
    acc = (torch.argmax(goal, dim=-1)==torch.argmax(guess, dim=-1)).float().mean()
    return acc.item()

def acc_regression (goal, guess):
    acc = torch.sqrt(((goal-guess)**2)).mean()
    return acc.item()

def polyak(a, b, alfa=0.99):
    for namea, parama in a.named_parameters():
        for nameb, paramb in b.named_parameters():
            if namea == nameb:
                paramb.data = paramb.data*alfa + parama.data*(1-alfa)
    return b 

def save(model, name, major, minor):
    PATH = 'pretrained_models/' +name+'_'+str(major)+'_'+str(minor)+'.th'
    torch.save(model.state_dict(), PATH)
def load(model, name, major, minor):
    PATH = 'pretrained_models/' +name+'_'+str(major)+'_'+str(minor)+'.th'
    model.load_state_dict(torch.load(PATH))
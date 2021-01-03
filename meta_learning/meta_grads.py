import copy
import torch
from meta_learning.utils import acc_classification, acc_regression, polyak

def maml_grad(model, intrain, targtrain, intest, targtest, lr, classification=True, first_order=False, device=torch.device('cpu')):
    params = list(model.parameters())
    return_grads = None
    batch_size = len(intrain)
    cumloss = 0
    cumacc = 0
    intrain, targtrain, intest, targtest = intrain.to(device), targtrain.to(device), intest.to(device), targtest.to(device)
    model = model.to(device)

    if classification:
        loss_func = lambda y, p: -(y*torch.log(p+1e-8)).mean()
        acc = acc_classification
    else:
        loss_func = lambda y, p: ((y-p)**2).mean()
        acc = acc_regression

    for x_train, y_train, x_test, y_test in zip(intrain, targtrain, intest, targtest):

        model1 = copy.deepcopy(model)
        preds = model1(x_train)
        loss1 = loss_func(y_train, preds)
        loss1.backward(create_graph=True, retain_graph=True)

        model2 = copy.deepcopy(model)
        news = []
        for param1, param2 in zip(model1.parameters(), model2.parameters()):
            tmp = param1 - lr*param1.grad 
            param2.data.copy_(tmp)
            news.append(tmp)

        preds = model2(x_test)
        loss = loss_func(y_test, preds)/batch_size
        cumacc += acc(y_test, preds)
        cumloss += loss.item()
        grads1 = torch.autograd.grad(loss, list(model2.parameters()))
        if first_order:
            grads2 = grads1
        else:
            grads2 = torch.autograd.grad(news, list(model1.parameters()), grad_outputs=grads1)
        
        if return_grads is None:
            return_grads = list(grads2)
        else:
            for i, g2 in enumerate(grads2):
                return_grads[i] += g2
    for p, g in zip(model.parameters(), return_grads):
        with torch.no_grad():
            if p.grad is None:
                p.grad = g
                continue
            p.grad += g
    return cumloss, cumacc/batch_size


def reptile_optimize(model, intrain, targtrain, intest, targtest, lr, n, alfa=0.95, update_model=True, classification=True, device=torch.device('cpu')):
    if classification:
        loss_func = lambda y, p: -(y*torch.log(p+1e-8)).mean()
        acc = acc_classification
    else:
        loss_func = lambda y, p: ((y-p)**2).mean()
        acc = acc_regression
    intrain, targtrain, intest, targtest = intrain.to(device), targtrain.to(device), intest.to(device), targtest.to(device)
    model = model.to(device)

    cumloss, cummacc = 0, 0
    for x_train, y_train, x_test, y_test in zip(intrain, targtrain, intest, targtest):
        model_copy = copy.deepcopy(model)
        opt = torch.optim.Adam(model_copy.parameters(), lr=lr)
        for i in range(n):
            preds = model_copy(x_train)
            loss = loss_func(y_train, preds)
            if i == 2:
                cumloss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        if update_model:
            polyak(model_copy, model, alfa=alfa)
        preds = model_copy(x_test)
        cummacc += acc(y_test, preds)
    return cumloss/intrain.shape[0], cummacc/intrain.shape[0], preds



import torch
import torch.nn.functional as F


def not_my_maml_grad(model, inputs, outputs, lr, batch=1):
    """
    Update a model's gradient using MAML.
    The gradient will point in the direction that
    improves the total loss across all inner-loop
    mini-batches.
    
    Args:
        model: an nn.Module for training.
        inputs: a large batch of model inputs.
        outputs: a large batch of model outputs.
        lr: the inner-loop SGD learning rate.
        batch: the inner-loop batch size.
    """
    params = list(model.parameters())
    device = params[0].device
    initial_values = []
    final_values = []
    losses = []
    scalar_losses = []

    for i in range(0, inputs.shape[0], batch):
        x = inputs[i:i+batch]
        y = outputs[i:i+batch]
        target = y.to(device)
        out = model(x.to(device))

        loss = -(outputs*torch.log(out+1e-8)).mean()
        losses.append(loss)
        scalar_losses.append(loss.item())
        initial_values.append([p.clone().detach() for p in params])
        updated = []
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        for grad, param in zip(grads, params):
            x = param - lr * grad
            updated.append(x)
            param.data.copy_(x)
        final_values.append(updated)

    gradient = [torch.zeros_like(p) for p in params]
    for loss, initial, final in list(zip(losses, initial_values, final_values))[::-1]:
        for p, x in zip(params, initial):
            p.data.copy_(x)
        grad1 = torch.autograd.grad(loss, params, retain_graph=True)
        grad2 = torch.autograd.grad(final, params, grad_outputs=gradient, retain_graph=True)
        gradient = [v1 + v2 for v1, v2 in zip(grad1, grad2)]

    for p, g in zip(params, gradient):
        if p.grad is None:
            p.grad = g
        else:
            p.grad.add_(g)
            
    return scalar_losses
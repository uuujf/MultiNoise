import os
from datetime import datetime
import torch
from torch.autograd import grad
import torch.nn.functional as F

def CEwithMask(input, target, mask=None):
    """mask should have identity mean"""
    input = F.log_softmax(input, dim=-1)
    bs = target.shape[0]
    loss = - input[range(bs), target]
    if mask is not None:
        loss = loss * mask
    loss = loss.mean()
    return loss

def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))

    if len(res) == 1:
        res = res[0]

    return res

class LogSaver(object):
    def __init__(self, logdir, logfile=None):
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if logfile:
            self.saver = os.path.join(logdir, logfile)
        else:
            self.saver = os.path.join(logdir, str(datetime.now())+'.log')
        print('save logs at:', self.saver)

    def save(self, item, name=None):
        with open(self.saver, 'a') as f:
            if name:
                f.write('======'+name+'======\n')
                print('======'+name+'======')
            f.write(item+'\n')
            print(item)

def evalFisherDiag(eval_list, model, criterion, optimizer):
    model.eval()

    diagF_dict = {}
    for name, param in model.named_parameters():
        diagF_dict[name] = torch.zeros_like(param)

    for x, y in eval_list:
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            diagF_dict[name] += param.grad.data.clone().pow(2)

    diag2 = 0
    for name in diagF_dict.keys():
        diagF_dict[name] /= len(eval_list)
        diag2 += torch.sum(diagF_dict[name])

    return diagF_dict, diag2

def evalCovDiag(eval_list, model, criterion, optimizer):
    model.eval()

    grads_dict = {}
    for name, _ in model.named_parameters():
        grads_dict[name]= []

    for x, y in eval_list:
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        for name, param in model.named_parameters():
            grads_dict[name].append(param.grad.data.clone())

    grad_std_dict = {}
    diag2 = 0
    for name in grads_dict.keys():
        grads_dict[name] = torch.stack(grads_dict[name], dim=0)
        grad_std_dict[name] = torch.std(grads_dict[name], dim=0)
        std_norm = torch.norm((grad_std_dict[name]).view(-1), p=2)
        diag2 += std_norm ** 2

    return grad_std_dict, diag2

def evalHessEigen(eval_tuple, model, criterion, n_iter=10):
    x, y = eval_tuple
    out = model(x)
    loss = criterion(out, y)
    g_tuple = grad(loss, model.parameters(), retain_graph=True, create_graph=True)

    v_list = [torch.randn_like(w) for w in model.parameters()]
    for i in range(n_iter-1):
        v_norm = torch.sqrt(sum([torch.sum(v.pow(2).view(-1)) for v in v_list]))
        gv_list = [torch.sum((g*v).view(-1))/(v_norm+1e-8) for g,v in zip(g_tuple, v_list)]
        v_list = list(grad(gv_list, model.parameters(), retain_graph=True)) # retain graph
    # release the graph
    v_norm = torch.sqrt(sum([torch.sum(v.pow(2).view(-1)) for v in v_list]))
    gv_list = [torch.sum((g*v).view(-1))/(v_norm+1e-8) for g,v in zip(g_tuple, v_list)]
    Hv_list = list(grad(gv_list, model.parameters())) # release graph
    Hv_norm = torch.sqrt(sum([torch.sum(v.pow(2).view(-1)) for v in Hv_list]))
    for v in Hv_list:
        v /= (Hv_norm+1e-8)
    return Hv_list


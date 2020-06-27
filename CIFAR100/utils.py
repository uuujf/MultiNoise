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

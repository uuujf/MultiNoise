import os
import argparse
import numpy as np
import torch
from tensorboardX import SummaryWriter

from cifar10 import CIFAR10
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--iters', type=int, default=int(1e5+1))
parser.add_argument('--schedule', type=int, nargs='+', default=[int(4e4), int(6e4)])
parser.add_argument('--batchsize', type=int, default=1000)
parser.add_argument('--ghostsize', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weightdecay', type=float, default=5e-4)
parser.add_argument('--aug', action='store_true', default=False)
parser.add_argument('--model', type=str, default='vgg')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--datadir', type=str, default='/mnt/home/haoyi/jingfengwu/datasets/CIFAR10/numpy')
parser.add_argument('--logdir', type=str, default='logs/SGD')

args = parser.parse_args()
logger = LogSaver(args.logdir)
logger.save(str(args), 'args')

# data
dataset = CIFAR10(args.datadir)
logger.save(str(dataset), 'dataset')
test_list = dataset.getTestList(500, True)

# model
start_iter = 0
lr = args.lr
if args.model == 'resnet':
    from resnet import ResNet18
    model = ResNet18().cuda()
elif args.model == 'vgg':
    from vgg import vgg11
    model = vgg11().cuda()
else:
    raise NotImplementedError()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, weight_decay=args.weightdecay)
if args.resume:
    checkpoint = torch.load(args.resume)
    start_iter = checkpoint['iter'] + 1
    lr = checkpoint['lr']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    logger.save("=> loaded checkpoint '{}'".format(args.resume))
logger.save(str(model), 'classifier')
logger.save(str(optimizer), 'optimizer')

# writer
writer = SummaryWriter(args.logdir)

# optimization
torch.backends.cudnn.benchmark = True
for i in range(start_iter, args.iters):
    # decay lr
    if i in args.schedule:
        lr *= 0.1
        logger.save('update lr: %f'%(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # train
    model.train()
    optimizer.zero_grad()
    ghost_list = dataset.getTrainGhostBatch(args.batchsize, args.ghostsize, args.aug, True)
    train_loss, train_acc = 0, 0
    for x,y in ghost_list:
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        train_acc += accuracy(out, y).item()
        train_loss += loss.item()
    for param in model.parameters():
        param.grad.data /= len(ghost_list)
    optimizer.step()
    train_acc /= len(ghost_list)
    train_loss /= len(ghost_list)

    # evaluate
    if i % 100 == 0 or i <= 100:
        writer.add_scalar('lr', lr, i)
        model.eval()
        out = model(x)
        train_acc = accuracy(out, y).item()
        train_loss = criterion(out, y).item()
        writer.add_scalar('acc/train', train_acc, i)
        writer.add_scalar('loss/train', train_loss, i)

        test_loss, test_acc = 0, 0
        for x,y in test_list:
            out = model(x)
            test_loss += criterion(out, y).item()
            test_acc += accuracy(out, y).item()
        test_loss /= len(test_list)
        test_acc /= len(test_list)
        writer.add_scalar('loss/test', test_loss, i)
        writer.add_scalar('acc/test', test_acc, i)
        writer.add_scalar('acc/diff', train_acc - test_acc, i)

        logger.save('Iter:%d, Test [acc: %.2f, loss: %.4f], Train [acc: %.2f, loss: %.4f]' \
                % (i, test_acc, test_loss, train_acc, train_loss))

    if i % 10000 == 0:
        state = {'iter':i, 'lr':lr, 'model':model.state_dict(), 'optimizer':optimizer.state_dict()}
        torch.save(state, args.logdir+'/iter-'+str(i)+'.pth.tar')

writer.close()

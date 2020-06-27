import os
import numpy as np
import torch

class SVHN(object):
    def __init__(self, data_dir):
        super(SVHN, self).__init__()
        self.n_classes = 10

        train = np.load(os.path.join(data_dir, 'train.npz'))
        test = np.load(os.path.join(data_dir, 'test.npz'))
        self.X_train = train['image']
        self.Y_train = train['label']
        self.X_test = test['image']
        self.Y_test = test['label']

        self.n_test = len(self.Y_test)
        self.n_train = len(self.Y_train)

        self.trans()

    def __str__(self):
        return 'SVHN\nnum_train: %d\nnum_test: %d' % (self.n_train, self.n_test)

    def transpose(self):
        self.X_train = np.transpose(self.X_train, [0,3,1,2])
        self.X_test = np.transpose(self.X_test, [0,3,1,2])

    def normalize(self):
        self.X_train = self.X_train / 255.0
        self.X_test = self.X_test / 255.0

    def trans(self):
        self.transpose()
        self.normalize()

    def to_tensor(self, X, Y, cuda=True):
        X = torch.FloatTensor(X)
        Y = torch.LongTensor(Y)
        if cuda:
            X = X.cuda()
            Y = Y.cuda()
        return X, Y

    def getTrainBatch(self, batch_size, cuda=True):
        mask = np.random.choice(self.n_train, batch_size, False)
        X = torch.FloatTensor(self.X_train[mask])
        Y = torch.LongTensor(self.Y_train[mask])
        return self.to_tensor(X, Y, cuda)

    def getTrainBatchList(self, batch_size, n_batch=100, cuda=True):
        batch_list = []
        for i in range(n_batch):
            X, Y = self.getTrainBatch(batch_size, cuda)
            batch_list.append((X, Y))
        return batch_list

    def getTestList(self, batch_size=5000, cuda=True):
        n_batch = self.n_test // batch_size
        batch_list = []
        for i in range(n_batch):
            X, Y = self.X_test[batch_size*i:batch_size*(i+1)], self.Y_test[batch_size*i:batch_size*(i+1)]
            X, Y = self.to_tensor(X, Y, cuda)
            batch_list.append((X, Y))
        return batch_list

    def getTrainList(self, batch_size=5000, cuda=False):
        n_batch = self.n_train // batch_size
        batch_list = []
        for i in range(n_batch):
            X, Y = self.X_train[batch_size*i:batch_size*(i+1)], self.Y_train[batch_size*i:batch_size*(i+1)]
            X, Y = self.to_tensor(X, Y, cuda)
            batch_list.append((X, Y))
        return batch_list

    def getTrainGhostBatch(self, batch_size=1000, ghost_size=100, cuda=True):
        mask = np.random.choice(self.n_train, batch_size, False)
        X, Y = self.X_train[mask], self.Y_train[mask]
        n_batch = batch_size // ghost_size
        ghost_list = []
        for i in range(n_batch):
            x, y = X[ghost_size*i:ghost_size*(i+1)], Y[ghost_size*i:ghost_size*(i+1)]
            x, y = self.to_tensor(x, y, cuda)
            ghost_list.append((x, y))
        return ghost_list

if __name__ == '__main__':
    datapath = '/Users/wjf/datasets/SVHN/train25k_test70k'
    dataset = SVHN(datapath)
    from IPython import embed; embed()

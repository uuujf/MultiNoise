import os
import numpy as np
import torch

class FashionMNIST(object):
    def __init__(self, data_dir):
        super(FashionMNIST, self).__init__()
        self.n_classes = 10

        train = np.load(os.path.join(data_dir, 'train.npz'))
        # clean = np.load(os.path.join(data_dir, 'clean.npz'))
        test = np.load(os.path.join(data_dir, 'test.npz'))

        self.X_train = train['image'].reshape(-1,28,28,1)
        self.Y_train = train['label']
        # self.mark = train['mark']
        # self.X_clean = clean['image']
        # self.Y_clean = clean['label']
        self.X_test = test['image'].reshape(-1,28,28,1)
        self.Y_test = test['label']

        self.n_test = len(self.Y_test)
        self.n_train = len(self.Y_train)
        # self.n_clean = len(self.Y_clean)

        self.trans()

    def __str__(self):
        # return 'FashionMNIST\nnum_train: %d\nnum_clean: %d' % (self.n_train, self.n_clean)
        return 'FashionMNIST\nnum_train: %d\nnum_test: %d' % (self.n_train, self.n_test)

    def transpose(self):
        self.X_train = np.transpose(self.X_train, [0,3,1,2])
        # self.X_clean = np.transpose(self.X_clean, [0,3,1,2])
        self.X_test = np.transpose(self.X_test, [0,3,1,2])

    def trans(self):
        self.transpose()

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

    def getTrainBatchList(self, batch_size, n_batch=20, cuda=True):
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

    def getTrainList(self, batch_size=1000, cuda=False):
        n_batch = self.n_train // batch_size
        batch_list = []
        for i in range(n_batch):
            X, Y = self.X_train[batch_size*i:batch_size*(i+1)], self.Y_train[batch_size*i:batch_size*(i+1)]
            X, Y = self.to_tensor(X, Y, cuda)
            batch_list.append((X, Y))
        return batch_list

if __name__ == '__main__':
    datapath = '/mnt/home/jingfeng/datasets/FashionMNIST/cl1000'
    fashion = FashionMNIST(datapath)
    from IPython import embed; embed()

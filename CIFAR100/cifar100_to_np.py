import argparse
import gzip
import os
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save2npz(images, labels, out_file):
    assert len(images) == len(labels)
    np.savez(out_file, image=images, label=labels)
    print('Save data to %s'%(out_file))
    return True

if __name__ == '__main__':
    n_examples = 10000
    height, width, channel = 32, 32, 3
    HOME = os.environ['HOME']
    DATASET = os.path.join(HOME, 'jingfengwu/datasets/CIFAR100')
    TARGET = os.path.join(DATASET, 'numpy')
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)

    # convet training data
    print('read train files')
    val_raw = unpickle(os.path.join(DATASET, 'cifar-100-python/train'))
    images = val_raw[b'data'].reshape(5*n_examples, channel, height, width)
    images = np.swapaxes(images, 1, 2)
    images = np.swapaxes(images, 2, 3)
    labels = np.array(val_raw[b'fine_labels'], dtype=np.uint8)
    print('convert train files')
    save2npz(images, labels, os.path.join(TARGET, 'train.npz'))

    # convet validation data
    print('read val files')
    val_raw = unpickle(os.path.join(DATASET, 'cifar-100-python/test'))
    images = val_raw[b'data'].reshape(n_examples, channel, height, width)
    images = np.swapaxes(images, 1, 2)
    images = np.swapaxes(images, 2, 3)
    labels = np.array(val_raw[b'fine_labels'], dtype=np.uint8)
    print('convert val files')
    save2npz(images, labels, os.path.join(TARGET, 'test.npz'))

    # test
    train = np.load(os.path.join(TARGET, 'train.npz'))
    test = np.load(os.path.join(TARGET, 'test.npz'))
    from IPython import embed; embed()

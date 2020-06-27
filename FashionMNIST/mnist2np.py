import os
import gzip
import numpy as np

def save2npz(images, labels, out_file, marks=None):
    assert len(images) == len(labels)
    if marks is None:
        np.savez(out_file, image=images, label=labels)
    else:
        np.savez(out_file, image=images, label=labels, mark=marks)
    print('Save data to %s'%(out_file))
    return True

if __name__ == '__main__':
    HOME = os.environ['HOME']
    DATASET = os.path.join(HOME, 'datasets/FashionMNIST')
    RAW = os.path.join(DATASET, 'raw_gz')
    # TARGET = os.path.join(DATASET, 'cl1000cr200')
    TARGET = os.path.join(DATASET, 'cl1000')
    if not os.path.exists(TARGET):
        os.makedirs(TARGET)

    filename = [
        ["training_images", "train-images-idx3-ubyte.gz"],
        ["test_images", "t10k-images-idx3-ubyte.gz"],
        ["training_labels", "train-labels-idx1-ubyte.gz"],
        ["test_labels", "t10k-labels-idx1-ubyte.gz"]
    ]

    mnist = {}
    for name in filename[:2]:
        with gzip.open(os.path.join(RAW, name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1,28*28)
    for name in filename[-2:]:
        with gzip.open(os.path.join(RAW, name[1]), 'rb') as f:
            mnist[name[0]] = np.frombuffer(f.read(), np.uint8, offset=8)

    # save test
    save2npz(mnist["training_images"], mnist["training_labels"], os.path.join(TARGET, 'test.npz'))
    # save2npz(mnist["test_images"], mnist["test_labels"], os.path.join(TARGET, 'test.npz'))

    # save train
    # process image
    CLEAN = 1000
    # CORRUPT = 0
    # images = mnist["training_images"]
    # labels = mnist["training_labels"]
    images = mnist["test_images"]
    labels = mnist["test_labels"]

    save2npz(images[0:CLEAN], labels[0:CLEAN], os.path.join(TARGET, 'train.npz'))

    # rnd_labels = np.random.choice(10, CORRUPT)
    # labels_fake = labels.copy()
    # labels_fake[CLEAN:CLEAN+CORRUPT] = rnd_labels
    # marks = (labels != labels_fake).astype(np.uint8)
    # save2npz(images[0:CLEAN+CORRUPT], labels_fake[0:CLEAN+CORRUPT], os.path.join(TARGET, 'train.npz'), marks[0:CLEAN+CORRUPT])
    # save2npz(images[0:CLEAN], labels_fake[0:CLEAN], os.path.join(TARGET, 'clean.npz'))


    # test the code
    train = np.load(os.path.join(TARGET, 'train.npz'))
    test = np.load(os.path.join(TARGET, 'test.npz'))
    from IPython import embed; embed()

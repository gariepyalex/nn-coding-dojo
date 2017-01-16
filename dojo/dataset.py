from __future__ import print_function
from urllib2 import urlopen, URLError, HTTPError
import os
import tarfile
import pickle
import numpy as np
import random

DATASET_PATH = './dataset/'

class Cifar10Dataset:

    def __init__(self):
        self.path = os.path.join(DATASET_PATH, 'cifar10/')
        self.download_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

        if not self._is_downloaded():
            self._download_dataset()

        self.x_train, self.y_train, self.x_test, self.y_test = self._load_dataset()


    def _is_downloaded(self):
        return os.path.isdir(self.path)


    def _download_dataset(self):
        print("Dowloading CIFAR10 dataset...")

        os.makedirs(self.path)
        tar_filename = os.path.basename(self.download_url)
        tar_filepath = os.path.join(self.path, tar_filename)
        dataset_file = urlopen(self.download_url)

        with open(tar_filepath, 'wb') as tar:
            tar.write(dataset_file.read())

        with tarfile.open(tar_filepath, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isreg():
                    member.name = os.path.basename(member.name)
                    tar.extract(member, self.path)

        os.remove(tar_filepath)
        print("Done.")


    def _load_cifar_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = pickle.load(f)
            x = datadict['data']
            y = datadict['labels']
            x = x.reshape(10000, 3, 32, 32).transpose(0,2,3,1).astype("float")
            y = np.array(y)
            return x, y


    def _load_dataset(self):
        xs = []
        ys = []
        for batch_number in range(1,6):
            batch_file = os.path.join(self.path, 'data_batch_%d' % batch_number)
            x, y = self._load_cifar_batch(batch_file)
            xs.append(x)
            ys.append(y)
        x_train = np.concatenate(xs)
        y_train = np.concatenate(ys)
        del x, y
        x_test, y_test = self._load_cifar_batch(os.path.join(self.path, 'test_batch'))
        return x_train, y_train, x_test, y_test



class PolynomialDataset:
    """
    Data in coming from a polynomial function of the shape y = ax^2 + bx + c + sigma, where a, b and c
    are parameters of the function and sigma is random noise of the distribution.
    """

    def __init__(self, n_of_points=1000, min_param=0, max_param=10):
        self.a = random.randint(min_param, max_param)
        self.b = random.randint(min_param, max_param)
        self.c = random.randint(min_param, max_param)
        self.sigma = 0.001

        xs = np.random.standard_normal(n_of_points)
        ys = self.a * (xs ** 2) + self.b * xs + self.c + random.random() * self.sigma

        split_index = 4 * n_of_points // 5

        self.x_train = xs[:split_index]
        self.y_train = ys[:split_index]

        self.x_test = xs[split_index:]
        self.y_test = ys[split_index:]



if __name__ == '__main__':
    p = PolynomialDataset()
    print(p.a, p.b, p.c)
    print(p.x_train[:10])
    print(p.y_train[:10])

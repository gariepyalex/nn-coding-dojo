import os
import tarfile
from urllib.request import urlopen, URLError, HTTPError

DATASET_PATH = './dataset/'

class Cifar10Dataset:

    def __init__(self):
        self.path = os.path.join(DATASET_PATH, 'cifar10/')
        self.download_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

        if not self._is_downloaded():
            self._download_dataset()


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

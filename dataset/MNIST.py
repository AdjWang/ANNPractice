from os import path, mkdir
from urllib import request
from multiprocessing.dummy import Pool as ThreadPool

source = 'http://yann.lecun.com/exdb/mnist/'
dataset = [
    'train-images-idx3-ubyte.gz',
    'train-labels-idx1-ubyte.gz',
    't10k-images-idx3-ubyte.gz',
    't10k-labels-idx1-ubyte.gz',
]


def download(address):
    return request.urlopen(address, timeout=30.0).read()


def download_all(output_path, dataset=dataset):
    address = [source+file for file in dataset]
    for addr in address:
        print(f'download: {addr}')
    # download to memory, takes about 11M
    p = ThreadPool(len(dataset))
    bin_dataset = p.map(download, address)
    # write to disk
    for content, file in zip(bin_dataset, dataset):
        with open(path.join(output_path, file), 'wb') as f:
            f.write(content)


def verify(mnist_path):
    if not path.exists(mnist_path):
        mkdir(mnist_path)
        download_all(mnist_path)
    else:
        pathes = [(i, path.join(mnist_path, i)) for i in dataset]
        to_download = [file for file, filepath in pathes
                       if not path.exists(filepath)]
        if len(to_download) > 0:
            download_all(mnist_path, to_download)

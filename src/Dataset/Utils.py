
from torch.utils.data import DataLoader
import struct
import gzip
import numpy as np
import random
import torch
import os
def getDataLoader(datasetType, dataset, batchSize, shuffle):

    dataLoader = DataLoader(dataset=datasetType(dataset),
                            batch_size=batchSize,
                            shuffle=shuffle, drop_last=True)
    return dataLoader

def logDataLoaderInfo(dataLoader):
    a,b,c,d = 0,0,0,0
    for elem in dataLoader.dataset.dataset:
        if elem[1] == 0 and elem[2]==0:
            a += 1
        elif elem[1] == 0 and elem[2]==1:
            b += 1
        elif elem[1] == 1 and elem[2]==0:
            c += 1
        elif elem[1] == 1 and elem[2]==1:
            d += 1
    print(a,b,c,d)

def _load_uint8(f):
    _, ndim = struct.unpack('BBBB', f.read(4))[2:]
    shape = struct.unpack('>' + 'I' * ndim, f.read(4 * ndim))
    buffer_length = int(np.prod(shape))
    data = np.frombuffer(f.read(buffer_length), dtype=np.uint8).reshape(shape)
    return data


def load_idx(path: str) -> np.ndarray:
    open_fcn = gzip.open if path.endswith('.gz') else open
    with open_fcn(path, 'rb') as f:
        return _load_uint8(f)

def load_file(path):
        return np.load(path).astype(np.float32)


def tensor_cuda(nr):
    return torch.tensor(nr, dtype=torch.long).cuda()

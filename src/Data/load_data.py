import torchvision
import torchvision.transforms as T
import os
import torch
import json
from Dataset.Utils import load_idx


def load_data_MorphoMNIST(usedDataset='global'):
    """ Download data from https://github.com/dccastro/Morpho-MNIST
    Loads the dataset from file "Data/Paths.json". The datasetType defines if train or test dataset is loaded.

    Args:
        usedDataset (str): Decides which dataset is used. 'global' means the original MNIST. Defaults to 'global'.
    Returns:
        dataset (dict): returns the dataset in a dict. With Sample, Label digit, Label Pert.
    """

    # Get the folder paths of the datasets
    with open(os.path.expanduser('~')+'/MIMM/src/Data/MorphoMNIST/Paths.json') as f:
        paths = json.load(f)
    datasetTypes = ["train", "test"]
    nrSamples = [60000, 10000]
    datasets = [{}, {}]
    for i, (datasetType, nr, dataset) in enumerate(zip(datasetTypes, nrSamples, datasets)):
        # Save dataset as dict, each dataset consits of data, label (number) and pertlabel (writing style)
        dataset[datasetType+"Data"] = torch.FloatTensor(
            load_idx(os.path.expanduser('~')+paths[usedDataset+"_"+datasetType+"Path"]).reshape(nr, 1, 28, 28))
        dataset[datasetType+"DataLabel"] = torch.LongTensor(
            load_idx(os.path.expanduser('~')+paths[usedDataset+"_"+datasetType+"PathLabel"]))

        # Add the perturbation label, if global use the given pert label, if only plain: 0, if only thin: 1, if only thick: 2
        dataset[datasetType+"PertDataLabel"] = torch.LongTensor(
            load_idx(os.path.expanduser('~')+paths[usedDataset+"_"+datasetType+"PertPathLabel"]))
        datasets[i] = dataset

    # Return trainData, testData
    return datasets[0], datasets[1]


def load_data_FashionMNIST():
    """ Download data from https://github.com/zalandoresearch/fashion-mnist """
    trainData = torchvision.datasets.FashionMNIST(os.path.expanduser('~') +
                                                  "/MIMM/Dataset", download=True, transform=T.Compose([T.ToTensor()]))
    testData = torchvision.datasets.FashionMNIST(os.path.expanduser('~') +
                                                 "/MIMM/Dataset", download=True, train=False, transform=T.Compose([T.ToTensor()]))

    return trainData, testData

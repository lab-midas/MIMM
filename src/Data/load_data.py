import torchvision
import torchvision.transforms as T
import os 
import torch
import json
from Dataset.Utils import load_idx



    
def load_data_MorphoMNIST(usedDataset='global'):
    """Loads the dataset from file "Data/Paths.json". The datasetType defines if train or test dataset is loaded.
    Each data sample consists of 3 variables: image of digit, label of digit, label of perturbation. 
    Label of perturbation can be: 0: plain (no pertrubation), 1: thinner (number written thinner), 2: thicker (number written thicker).

    Args:
        datasetType (str): train or test Defaults to 'train'.
        usedDataset (str): Decides which dataset is used. 'global' means the original MNIST. Defaults to 'global'.
        nrDatasamples (int): train: 60000, test: 10000. Defaults to 60000.

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

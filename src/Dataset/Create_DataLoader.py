import sys
import os
sys.path.insert(0, os.getcwd()+'/MIMM/src')

from Data.load_data import load_data_MorphoMNIST
from Dataset.Create_Dataset import create_dataset
from torchvision import transforms
from torchvision.datasets import MNIST


def init_morphoMNIST_training(params_dict, training_dataset):
    # Load data and create loaders
    trainData, valData = load_data_MorphoMNIST()
    trainLoader, valLoader = create_dataset(trainData, valData, params_dict, training_dataset)
    _, testLoader = create_dataset("", valData, params_dict, training_dataset, valData=True)
    params_dict["confoundingRatio"]["value"] = 0.5
    _, testEqualLoader = create_dataset("", valData, params_dict, training_dataset, valData=True)
    params_dict["confoundingRatio"]["value"] = 0.95

    return trainLoader, valLoader, [testLoader], testEqualLoader

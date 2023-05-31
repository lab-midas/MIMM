
import torch
import random
from Dataset.Utils import getDataLoader, logDataLoaderInfo
from Dataset.Dataset import DatasetTwoLabels


def create_MorphoMNIST_dataset_confounded(datasetType, originalDataset, params_dict, rotation=0, batchSize = 10, shuffle=False):
    # Get the preset hyperparameters

    # two types of classes and confounding confounding
    classes_pt = list(range(2))
    classes_sc = [x for x in list(range(2))]

    # Extract data, original class labels (0-9), perturbed labels (writing style)
    dataset = originalDataset[datasetType + "Data"].cuda()
    datasetLabels = originalDataset[datasetType + "DataLabel"].cuda()
    datasetPertLabels = originalDataset[datasetType + "PertDataLabel"].cuda()

    # change the PerLabels, since 1: thin, 2: thick to 0:thin and 1:thick
    datasetPertLabels = [x-1 for x in datasetPertLabels]

    # create the new label CL (0, 1, 2) from the original numbers 0-9
    classGroupLabels = groupNumbersAsNewClassLabels_2_ptasses(datasetLabels)

    full_dataset = list(zip(dataset, classGroupLabels, datasetPertLabels))

    full_dataset = sorted(full_dataset, key=lambda x: x[2])
    full_dataset = sorted(full_dataset, key=lambda x: x[1])

    dataset_splitted_pt_and_sc = []
    for i in classes_pt:
        dataset_splitted_pt_and_sc.append([])
        for j in classes_sc:
            dataset_splitted_pt_and_sc[i].append([])

    for i, _ in enumerate(full_dataset):
        for pt_idx, pt in enumerate(classes_pt):
            for sc_idx, sc in enumerate(classes_sc):
                if full_dataset[i][1] == pt and full_dataset[i][2] == sc:
                    dataset_splitted_pt_and_sc[pt_idx][sc_idx].append(
                        full_dataset[i])

    minlen = min([len(dataset_splitted_pt_and_sc[0][0]), len(dataset_splitted_pt_and_sc[0][1]), len(dataset_splitted_pt_and_sc[1][0]), len(dataset_splitted_pt_and_sc[1][1])])
    for pt_idx, pt in enumerate(classes_pt):
        for sc_idx, sc in enumerate(classes_sc):
            dataset_splitted_pt_and_sc[pt_idx][sc_idx] = dataset_splitted_pt_and_sc[pt_idx][sc_idx][:minlen]

    if isinstance(params_dict["confoundingRatio"]["value"], list):
        non_confounding_ratio = (1-confounding_ratio)/(len(classes_sc)-1)

    elif params_dict["confoundingRatio"]["value"] != 0:
        confounding_ratio = params_dict["confoundingRatio"]["value"]
        non_confounding_ratio = (1-confounding_ratio)/(len(classes_sc)-1)
    else:
        confounding_ratio, non_confounding_ratio = 1

    confounded_dataset = []
    for pt_idx, pt in enumerate(classes_pt):
        for sc_idx, sc in enumerate(classes_sc):
            if (rotation == 0 and sc_idx == pt_idx) or ((rotation == 1) and (sc_idx == ((pt_idx+rotation) % len(dataset_splitted_pt_and_sc)))):
                nrSamples = int(
                    len(dataset_splitted_pt_and_sc[pt_idx][sc_idx])*confounding_ratio)
            else:
                nrSamples = int(
                    len(dataset_splitted_pt_and_sc[pt_idx][sc_idx])*non_confounding_ratio)
            dataset_splitted_pt_and_sc[pt_idx][sc_idx] = random.sample(
                dataset_splitted_pt_and_sc[pt_idx][sc_idx], nrSamples)
            confounded_dataset.extend(
                dataset_splitted_pt_and_sc[pt_idx][sc_idx])
    
    # Get dataloader
    dataLoader = getDataLoader(DatasetTwoLabels, confounded_dataset, batchSize, shuffle=shuffle)
    logDataLoaderInfo(dataLoader)
    return dataLoader


def groupNumbersAsNewClassLabels_2_ptasses(datasetLabels, counts=False):
    """
    Used to split the dataset from the original class label 0-9 to new class groups, where:
    class 0 contains the numbers 0,1,2, class 1 contains 3-6, class 2 contains 7-9

    Args:
        datasetLabels (list(int)): list of the original labels 0-9 of all images

    Returns:
        newDataLabel (list(int)): list of the new labels 0-2 of all images
    """
    newDataLabel = []
    smaller = 0
    greater = 0
    for label in datasetLabels:
        if label < 5:
            newDataLabel.append(0)
            smaller += 1
        elif label >= 5:
            newDataLabel.append(1)
            greater += 1
    newDataLabel = torch.LongTensor(newDataLabel).cuda()
    if counts:
        return newDataLabel, smaller, greater
    return newDataLabel

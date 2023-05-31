import random
from Dataset.Dataset import DatasetTwoLabels, DatasetOneLabel
from Dataset.Utils import getDataLoader, logDataLoaderInfoFM
import torch


def create_FM_dataset(originalDataset, params_dict):

    dataset = originalDataset.data.type(torch.FloatTensor)
    dataset = dataset.reshape((dataset.shape[0], 1, 28, 28)).cuda()
    datasetLabels = originalDataset.train_labels.cuda()
    full_dataset = list(zip(dataset, datasetLabels))
    # Get dataloader
    batchSize = params_dict["batchSize"]["value"]
    dataLoader = getDataLoader(
        DatasetOneLabel, full_dataset, batchSize, shuffle=True)

    return dataLoader


def create_FM_dataset_confounded(originalDataset, params_dict, rotation=0, valData=False):

    classes_pt = list(range(params_dict["numberClasses"]["value"]))
    classes_sc = list(range(params_dict["numberClasses"]["value"]))

    if params_dict["confoundingRatio"]["value"] != 0:
        confounding_ratio = params_dict["confoundingRatio"]["value"]
    else:
        confounding_ratio = 1
    
    dataset = originalDataset.data.type(torch.FloatTensor)
    dataset = dataset.reshape((dataset.shape[0], 1, 28, 28)).cuda()
    datasetLabels = originalDataset.train_labels.cuda()
    datasetPertLabels = datasetLabels.clone().cuda()
    full_dataset = list(
        map(list, zip(dataset, datasetLabels, datasetPertLabels)))
    full_dataset = sorted(full_dataset, key=lambda x: x[1])

    length_full_dataset = len(full_dataset)
    length_one_class = int(len(full_dataset)/10)
    # Create empty lists for each class
    dataset_splitted_pt = []
    for _ in classes_pt:
        dataset_splitted_pt.append([])

    dataset_splitted_pt_and_sc = []
    for i in classes_pt:
        dataset_splitted_pt_and_sc.append([])
        for j in classes_sc:
            dataset_splitted_pt_and_sc[i].append([])

    # Separate dataset by classes.
    for i, nr in enumerate(range(0, length_full_dataset, length_one_class)):
        dataset_splitted_pt[i] = full_dataset[nr:nr+length_one_class]

    # Separate the dataset by classes and confounding
    for pt in classes_pt:
        len_dataset_pt = len(dataset_splitted_pt[pt])
        nr_samples = int(len_dataset_pt*confounding_ratio)
        nr_samples_non_confounding = int(
            (len_dataset_pt-nr_samples)/(len(classes_sc)-1))
        init_nr_samples_non_confounding = nr_samples
        for sc in classes_sc:
            if pt == sc:
                dataset_splitted_pt_and_sc[pt][sc] = dataset_splitted_pt[pt][:nr_samples]
                for i, img in enumerate(dataset_splitted_pt_and_sc[pt][sc]):
                    dataset_splitted_pt_and_sc[pt][sc][i][0] = bars_images_confounding(
                        pt, img[0], rotation)
                    dataset_splitted_pt_and_sc[pt][sc][i][2] = torch.tensor(
                        (sc+rotation) % 10, dtype=torch.long).cuda()
                    dataset_splitted_pt_and_sc[pt][sc][i] = tuple(
                        dataset_splitted_pt_and_sc[pt][sc][i])
            else:
                dataset_splitted_pt_and_sc[pt][sc] = dataset_splitted_pt[pt][init_nr_samples_non_confounding:
                                                                             init_nr_samples_non_confounding+nr_samples_non_confounding]
                for i, img in enumerate(dataset_splitted_pt_and_sc[pt][sc]):
                    dataset_splitted_pt_and_sc[pt][sc][i][0] = bars_images_confounding(
                        sc, img[0], rotation)
                    dataset_splitted_pt_and_sc[pt][sc][i][2] = torch.tensor(
                        (sc+rotation) % 10, dtype=torch.long).cuda()

                    dataset_splitted_pt_and_sc[pt][sc][i] = tuple(
                        dataset_splitted_pt_and_sc[pt][sc][i])
                init_nr_samples_non_confounding += nr_samples_non_confounding

    if not valData and "RESAMPLE" in params_dict["trainType"]["value"]:
        dataset_splitted_pt_and_sc = overSampleDataWithResampling(
            dataset_splitted_pt_and_sc)

    # Merge to one dataset
    full_dataset_confounded = []
    for pt in classes_pt:
        for sc in classes_sc:
            full_dataset_confounded.extend(dataset_splitted_pt_and_sc[pt][sc])

    # Get dataloader
    batchSize = params_dict["batchSize"]["value"]
    dataLoader = getDataLoader(
        DatasetTwoLabels, full_dataset_confounded, batchSize, shuffle=True)
   # logDataLoaderInfoFM(dataLoader)

    return dataLoader


def overSampleDataWithResampling(dataConfounded):

    nrOfResamples = int(len(dataConfounded[0][0])/len(dataConfounded[0][1]))
    # Minor samples class 0
    balanced_dataset = []
    classes_pt, classes_sc = 10, 10
    for i in range(classes_pt):
        balanced_dataset.append([])
        for j in range(classes_sc):
            balanced_dataset[i].append([])
    for i in range(10):
        for j in range(10):
            if i != j:
                minor_samples = dataConfounded[i][j]*nrOfResamples+random.sample(
                    dataConfounded[i][j], len(dataConfounded[i][i]) - len(dataConfounded[i][j]*nrOfResamples))
                balanced_dataset[i][j] = minor_samples
            elif i == j:
                balanced_dataset[i][j] = dataConfounded[i][j]

    return balanced_dataset


def bars_images_confounding(pt, img, rotation=0):

    rot = ((pt+rotation) % 10)
    if 0 == rot:
        img[0, :, 0:3] = 255
    elif 1 == rot:
        img[0, :, -3:28] = 255
    elif 2 == rot:
        img[0, 0:3, :] = 255
    elif 3 == rot:
        img[0, -3:28, :] = 255
    elif 4 == rot:
        img[0, :, 0:3] = 255
        img[0, :, -3:28] = 255
    elif 5 == rot:
        img[0, 0:3, :] = 255
        img[0, -3:28, :] = 255
    elif 6 == rot:
        img[0, :, 0:3] = 255
        img[0, 0:3, :] = 255
    elif 7 == rot:
        img[0, -3:28, :] = 255
        img[0, :, -3:28] = 255
    elif 8 == rot:
        img[0, 0:3, :] = 255
        img[0, :, -3:28] = 255
    elif 9 == rot:
        img[0, :, 0:3] = 255
        img[0, -3:28, :] = 255
    return img

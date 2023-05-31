
from Dataset.Create_Dataset_MorphoMNIST import create_MorphoMNIST_dataset_confounded

def create_dataset(trainData, testData, params_dict, training_dataset, valData = False):
    """Create datasets.
    Args:
        trainData (_type_): training data
        testData (_type_): test data
        params_dict (_type_): parameter
        training_dataset (_type_): training dataset used. Type of experiment (FashionMNIST, MorphoMIST)
        rotation (int, optional): Needed for dataset creation during testing.
        valData (bool, optional): If false, trainData and valdata (for during training created.). If true, testData for evaluation.
    """
    
    if training_dataset == "MorphoMNIST":
        if not valData:
            trainLoader = create_MorphoMNIST_dataset_confounded('train', trainData, params_dict, rotation=0, batchSize=params_dict["batchSize"]["value"], shuffle=True)
            testLoader = create_MorphoMNIST_dataset_confounded('test', testData, params_dict, rotation=0, batchSize=10)
        else:
            trainLoader = ""
            testLoader = create_MorphoMNIST_dataset_confounded('test',testData, params_dict, rotation=1, batchSize=10)
        return trainLoader, testLoader
        

    
import torch.nn as nn

def crossEntropyLoss_for_MTL(y_pred, y_true):
    """ Compute CrossEntropyLoss for multi-task learning

    Args:
        y_pred (list): list of predictions, each entry contains a list with computed predictions of one batch
        y_true (list): list of corresponding ground truth labels, each entry contains a list with gt of one batch

    Returns:
        loss_average[float]: returns loss averaged over the number of multi-tasks  
    """
    loss = 0
    criterionLoss = nn.CrossEntropyLoss()
    for i in range(len(y_pred)):
        loss += criterionLoss(y_pred[i], y_true[i])
    return loss/len(y_pred)

def crossEntropyLoss_for_STL(y_pred, y_true):
    """ Compute CrossEntropyLoss for a single task learning

    Args:
        y_pred (list): computed predictions of one batch
        y_true (list): list of corresponding ground truth labels

    Returns:
        loss_average[float]: returns loss 
    """

    criterionLoss = nn.CrossEntropyLoss()
    return criterionLoss(y_pred, y_true)
     
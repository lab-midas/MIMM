import numpy as np
from sklearn import metrics

def compute_metrics(y_pred, y_true):
    """ Computes a set of classification metrics
        Accuracy
    Args:
        y_pred (torch.Tensor): A num_samples x num_ptasses tensor with the predicted probabilities per class. 
            Should sum to 1 per example.
        y_train (torch.Tensor]): A num_samples tensor (int or bool) with the groudn truth class.

    Returns:
        accuracy(float): Classification accuracy
    """
    # Turn into numpy elements on cpu
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    # Compute Metrics
    predicted = np.argmax(y_pred, -1)
    accuracy = metrics.accuracy_score(y_true, predicted)*100


    return accuracy

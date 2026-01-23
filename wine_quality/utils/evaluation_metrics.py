import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd

from typing import Union

def plot_confusion_matrix(
        y_test_tensor : Union[torch.Tensor,np.ndarray], 
        y_preds : Union[torch.Tensor,np.ndarray]
    ) -> None:
    r"""
    Plots confusion matrix for the predicted vs true labels

    Args:
        y_test_tensor (torch.Tensor) : True labels for the test set.
        y_preds (torch.Tensor) : Predicted labels from the model

    Returns:
        None
    """
    
    if isinstance(y_test_tensor, torch.Tensor):
        y_test_tensor = y_test_tensor.detach().cpu().numpy() 

    if isinstance(y_test_tensor, np.ndarray):
        y_preds = y_preds.detach().cpu().numpy() 

    cm = confusion_matrix(y_test_tensor, y_preds)

    # Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix: Wine Quality')
    plt.show()

def accuracy(
        y_logits : torch.Tensor, 
        y_test : torch.Tensor
    ) -> float:
    r"""
    Compute the classification accuracy for given model outputs

    This function applies a softmax to the logits, selects the predicted class
    with the highest probability, and compares it to the true labels.

    Args:
        y_logits (torch.Tensor) : logits (raw scores) of size (batch_size, num_classes)
        y_test (torch.Tensor) : Ground truth labels of shape (batch_size,).

    Returns:
        Accuracy - correct predictions/total predictions
    """

    preds = torch.softmax(y_logits, dim=1).argmax(dim=1)
    accuracy = (preds == y_test).float().mean().item()

    return accuracy

def plot_correlation_matrix(data : pd.DataFrame) -> None:
    r"""
    Plots and shows correlation matrix for the features in the given dataframe

    Args:
        data (pd.DataFrame)
    
    Returns:
        None
    """

    corr = data.corr()

    plt.imshow(corr, cmap='coolwarm', interpolation='none')

    plt.colorbar()
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    plt.title('Correlation Matrix')
    plt.show()
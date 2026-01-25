import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.logging_utils import create_log, plot_training_results
from logger import get_logger
import torch.optim as optim
from ray import tune
from utils.evaluation_metrics import accuracy
import os
import tempfile
from ray.tune import Checkpoint

from torch import Tensor
from typing import Tuple, Dict, Any, Type, Optional
from torch.optim import Optimizer
from torch import nn

logger = get_logger(__name__)


def train_dataset_v0(
            model  : nn.Module, # Change to specific class in case of accepting only that class and its children
            x_train_tensor : Tensor,
            y_train_tensor : Tensor,
            x_test_tensor : Tensor,
            y_test_tensor : Tensor,
            optimizer : Optimizer,
            loss_function : nn.CrossEntropyLoss, # Change to nn.Module in case of different type of loss - may require changes in the logic of the code.
            epochs : int,
            device : str,
            log_file : Optional[str] = None,
            plots_folder_path : str = None,
            batch_size : int = 32,
            is_tuning : bool = False,
            save_fig : bool = False,
        ) -> Tuple[nn.Module,float]:

    r""" Train a PyTorch model on the given dataset and evaluate performance. 
    
    This function performs training and evaluation over a specified number of epochs. 
    It logs training and test loss values, computes accuracy, and optionally integrates with Ray Tune for hyperparameter optimization.
    If not tuning, it saves the best model and generates plots/logs of training results.
    Also saves the plot if save_fig is True 
    
    Args: 
        model (nn.Module): The PyTorch model to train. 
        x_train_tensor (Tensor): Training features as a tensor. 
        y_train_tensor (Tensor): Training labels as a tensor. 
        x_test_tensor (Tensor): Test features as a tensor.
        y_test_tensor (Tensor): Test labels as a tensor. 
        optimizer (Optimizer): Optimizer instance (e.g., Adam, SGD). 
        loss_function (nn.CrossEntropyLoss): Loss function used for training. 
        epochs (int): Number of training epochs. 
        device (str): Device to run training on ("cpu" or "cuda"). 
        log_file (str): Path to the log file for saving training results. 
        plots_folder_path (str) : Path to the folder for saving plots
        batch_size (int, optional): Batch size for training. Defaults to 32. 
        is_tuning (bool, optional): If True, reports metrics to Ray Tune. Defaults to False. 
        save_fig (bool, optional) : If True, saves the plots generated. Defaults to False.
        
    Returns: 
        Tuple[nn.Module, float]: The trained model and the best accuracy achieved. 
    """
    if save_fig and not plots_folder_path and not is_tuning:
        raise ValueError( "plots_folder_path must be provided when save_fig=True and is_tuning=False. " 
                         "If you do not want to save plots, set save_fig=False." ) 

    if not log_file and not is_tuning:
        logger.warning("Log file path not specified. Not logging results to csv.")

 
    train_loss_values = []
    test_loss_values = []
    accuracy_values = []

    train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 

    for epoch in range(epochs):

        model.train()

        batch_loss = 0
        member_count = 0

        for x_batch, y_batch in train_loader:

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            y_logits = model(x_batch)

            loss = loss_function(y_logits, y_batch.long())
            batch_loss += loss.item()
            member_count += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        with torch.inference_mode():
            test_pred = model(x_test_tensor)

            test_loss_value = loss_function(test_pred, y_test_tensor.type(torch.long)).item()

            accuracy_metric = accuracy(y_logits = test_pred, y_test = y_test_tensor)

            accuracy_values.append(accuracy_metric)
            train_loss_values.append(batch_loss/ member_count)
            test_loss_values.append(test_loss_value)

            if is_tuning:
                tune.report({"accuracy":accuracy_metric,"loss":test_loss_value})
            else:
                if epoch % 10 == 0:
                    logger.info(f"Epoch : {epoch} \n\tTrain Loss : {batch_loss/ member_count} \n\tTest Loss : {test_loss_value} \n\tAccuracy : {accuracy_metric}\n")

    if not test_loss_values or not accuracy_values: 
        raise RuntimeError("No training iterations were completed. Check dataset size and epochs.")

    best_test_loss = min(test_loss_values)
    best_epoch = test_loss_values.index(best_test_loss)
    train_loss_at_best_test = train_loss_values[best_epoch]

    best_accuracy = max(accuracy_values)
    best_accuracy_epoch = accuracy_values.index(best_accuracy)

    if not is_tuning:
        if log_file:
            create_log(log_file,hyperparameters=model.__str__(),test_loss=best_test_loss,train_loss=train_loss_at_best_test,best_epoch=best_epoch,best_accuracy=best_accuracy, best_accuracy_epoch=best_accuracy_epoch)
        
        plot_training_results(
            train_loss_values=train_loss_values,
            test_loss_values=test_loss_values,
            accuracy_values=accuracy_values,
            save_fig=save_fig,
            fig_name=f"{model.__class__.__name__}_training_plot.png",
            folder_path=plots_folder_path)

    return model, best_accuracy



def ray_tune_wrapper(
        config : Dict[str,Any], 
        model_class : Type[nn.Module], 
        x_train_tensor : Tensor,
        y_train_tensor : Tensor,
        x_test_tensor : Tensor,
        y_test_tensor : Tensor,
        input_dim : int,
        output_dim :int
    ) -> None:
    r"""
    Wrapper function for Ray Tune to train and evaluate a model with given hyperparameters.

    This function initializes a model instance, optimizer, and loss function based on
    the provided configuration. It then calls `train_dataset_v0` with tuning enabled,
    allowing Ray Tune to collect accuracy and loss metrics for hyperparameter search.

    Args:
        config (Dict[str, Any]): Dictionary of hyperparameters (e.g., optimizer, lr, epochs, batch_size).
        model_class (Type[nn.Module]): Model class to instantiate for training.
        x_train_tensor (Tensor): Training features as a tensor.
        y_train_tensor (Tensor): Training labels as a tensor.
        x_test_tensor (Tensor): Test features as a tensor.
        y_test_tensor (Tensor): Test labels as a tensor.
        input_dim (int) : input layer features,
        output_dim (int) : output layer features

    Returns:
        None: Metrics are reported to Ray Tune; no explicit return value.
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device : {device}")

    tuning_model = model_class(input_dim=input_dim, output_dim=output_dim).to(device)
    
    optimizer_class = config.get("optimizer", optim.Adam)
    optimizer = optimizer_class(tuning_model.parameters(), lr=config.get("lr", 0.001))
    
    loss_function = torch.nn.CrossEntropyLoss()


    trained_model, best_accuracy = train_dataset_v0(
        model=tuning_model,
        x_train_tensor=x_train_tensor.to(device),
        y_train_tensor=y_train_tensor.to(device),
        x_test_tensor=x_test_tensor,
        y_test_tensor=y_test_tensor,
        optimizer=optimizer,
        loss_function=loss_function,
        epochs=config.get("epochs", 100),
        device=device,
        batch_size=config.get("batch_size", 32),
        is_tuning=True
    )
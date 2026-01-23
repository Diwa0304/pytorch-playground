import csv
import os
import matplotlib.pyplot as plt

from typing import List, Optional

def create_log(
        filename : str,
        hyperparameters : str,
        test_loss : float,
        train_loss : float,
        best_epoch : int,
        best_accuracy : float, 
        best_accuracy_epoch : int
    ) -> None:
    r"""
    Logs metrics and model's hyperparameters into log file

    Args:
        filename (str) : the path to the file to store log
        hyperparameters (str) : a string of model's architecture - obtained by model.__Str__ or manually formatted string
        test_loss (float) : the best test loss obtained
        train_loss (float) : the train loss of the corresponding best test loss
        best_epoch (int) : the epoch of the corresponding best test loss
        best_accuracy (float) : the best accuracy obtained over epochs
        best_accuracy_epoch (int) : the epoch of the corresponding best accuracy

    Returns:
        None
    """
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory): 
        os.makedirs(directory, exist_ok=True)
    
    if not os.path.exists(filename):
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "HyperParameters", 
                "best_test_loss", "train_loss_at_best_test", "best_epoch", "best_accuracy", "best_accuracy_epoch"
            ])
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            hyperparameters,
            test_loss,
            train_loss,
            best_epoch,
            best_accuracy,
            best_accuracy_epoch
        ])


def plot_training_results(
        train_loss_values : List[float],
        test_loss_values : List[float],
        accuracy_values : List[float],
        save_fig : bool = False ,
        fig_name : Optional[str] = None,
        folder_path : str = "results_figures"
    ) -> None:
    r"""
        plot 1 -> test loss and train loss against the epoch
        plot 2 -> accuracy against epoch
        plots them in a single figure and saves the figure(if mentioned) in the given folder

        Args:
            train_loss_values (list) : list of train losses over epochs
            test_loss_values (list) : list of test losses over epochs
            accuracy_values (list) : list of accuracy values over epochs
            save_fig (bool) : flag to save the plot figure
            folder_path (str) : path of the destination folder for the plot figure

        Returns:
            None
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                                             
    axes[0].plot(range(len(train_loss_values)), train_loss_values, label='Train Loss')    
    axes[0].plot(range(len(test_loss_values)), test_loss_values, label='Test Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train and Test Loss over Epochs')
    axes[0].legend()

    axes[1].plot(range(len(accuracy_values)),accuracy_values, label="Accuracy")
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].legend()

    plt.tight_layout()

    if save_fig:
        os.makedirs(folder_path, exist_ok=True)
        if fig_name:
            plt.savefig(os.path.join(folder_path, fig_name))
        else:
            plt.savefig(os.path.join(folder_path, "training_plot.png"))
    plt.show()

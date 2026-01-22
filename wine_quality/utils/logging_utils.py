import csv
import os
import matplotlib.pyplot as plt

def create_log(filename,hyperparameters,test_loss,train_loss,best_epoch):
    if not os.path.exists(filename):
        os.makedirs('results', exist_ok=True)
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "HyperParameters", 
                "best_test_loss", "train_loss_at_best_test", "best_epoch"
            ])
    with open(filename, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            hyperparameters,
            test_loss,
            train_loss,
            best_epoch
        ])


def plot_training_results(epoch_count,train_loss_values,test_loss_values,save_fig=False,fig_name=None,folder_path="results_figures"):
    
    plt.plot(epoch_count, train_loss_values, label='Train Loss')    
    plt.plot(epoch_count, test_loss_values, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss over Epochs')
    plt.legend()

    if os.path.exists(folder_path) == False and save_fig:
        os.makedirs(folder_path)

    if save_fig and fig_name:
        plt.savefig(os.path.join(folder_path, fig_name))
    elif save_fig and not fig_name:
        plt.savefig(os.path.join(folder_path, "training_plot.png"))
    plt.show()

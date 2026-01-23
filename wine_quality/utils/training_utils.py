import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.logging_utils import create_log, plot_training_results
from logger import get_logger
import torch.optim as optim
from ray import train, tune


logger = get_logger(__name__)


def train_dataset_v0(model,x_train_tensor,y_train_tensor,x_test_tensor,y_test_tensor,optimizer,loss_function,epochs,device,log_file,batch_size=32,is_tuning=False):
    epoch_count = []
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

            preds = torch.softmax(test_pred, dim=1).argmax(dim=1)
            accuracy = (preds == y_test_tensor).float().mean().item()

            accuracy_values.append(accuracy)
            epoch_count.append(epoch)
            train_loss_values.append(batch_loss/ member_count)
            test_loss_values.append(test_loss_value)

            if is_tuning:
                tune.report({"accuracy":accuracy,"loss":test_loss_value})
            else:
                if test_loss_value < 0.8618305325508118:
                    best_test_loss = test_loss_value
                    torch.save(model.state_dict(), "red_wine_models/best_model.pth")

                if epoch % 10 == 0:
                    logger.info(f"Epoch : {epoch} \n Train Loss : {batch_loss/ member_count} \n Test Loss : {test_loss_value} \nAccuracy : {accuracy}")


    best_test_loss = min(test_loss_values)
    best_epoch = test_loss_values.index(best_test_loss)
    train_loss_at_best_test = train_loss_values[best_epoch]

    best_accuracy = max(accuracy_values)
    best_accuracy_epoch = accuracy_values.index(best_accuracy)

    if not is_tuning:
        create_log(log_file,hyperparameters=model.__str__(),test_loss=best_test_loss,train_loss=train_loss_at_best_test,best_epoch=best_epoch,best_accuracy=best_accuracy, best_accuracy_epoch=best_accuracy_epoch)
        plot_training_results(epoch_count=epoch_count,train_loss_values=train_loss_values,test_loss_values=test_loss_values,accuracy_values=accuracy_values,save_fig=True,fig_name=f"{model.__class__.__name__}_training_plot.png")

    return model, best_accuracy



def ray_tune_wrapper(config, model_class, x_train_tensor,y_train_tensor,x_test_tensor,y_test_tensor):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tuning_model = model_class(input_dim=11, output_dim=6 ).to(device)
    
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
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_file="ray_tune/ray_tune_logs.txt",
        batch_size=config.get("batch_size", 32),
        is_tuning=True
    )
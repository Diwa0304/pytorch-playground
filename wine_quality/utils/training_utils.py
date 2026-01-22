import torch
from torch.utils.data import TensorDataset, DataLoader
from utils.logging_utils import create_log, plot_training_results
from logger import get_logger

logger = get_logger(__name__)


def train_dataset_v0(model,x_train_tensor,y_train_tensor,x_test_tensor,y_test_tensor,optimizer,loss_function,epochs,device,log_file,batch_size=32):
    epoch_count = []
    train_loss_values = []
    test_loss_values = []

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

            test_loss = loss_function(test_pred, y_test_tensor.type(torch.long))

            epoch_count.append(epoch)
            train_loss_values.append(batch_loss/ member_count)
            test_loss_values.append(test_loss.item())

            if test_loss < 0.8618305325508118:
                best_test_loss = test_loss
                torch.save(model.state_dict(), "red_wine_models/best_model.pth")
            
            if epoch % 10 == 0:
                logger.info(f"Epoch : {epoch} \n Train Loss : {batch_loss/ member_count} \n Test Loss : {test_loss.item()} \n")

    best_test_loss = min(test_loss_values)
    best_epoch = test_loss_values.index(best_test_loss)
    train_loss_at_best_test = train_loss_values[best_epoch]

    create_log(log_file,hyperparameters=model.__str__(),test_loss=best_test_loss,train_loss=train_loss_at_best_test,best_epoch=best_epoch)

    plot_training_results(epoch_count=epoch_count,train_loss_values=train_loss_values,test_loss_values=test_loss_values,save_fig=True,fig_name=f"{model.__class__.__name__}_training_plot.png")
    return model


import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import PIL
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import optuna
from tqdm import tqdm

# Constants
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
epochs = 15
stored_hp_exp = "sqlite:///optuna_trials_train_metrics/optuna_study.db"
save_folder = "final_model"
final_model = 'final_model.pt'
hp_study_name = "hyperparameter_search_mnist"
filename = "final_model_test_metrics.csv"
train_f_name = "final_model_train_loss_and_accuracy.csv"


def log_test_metrics(test_loss, test_accuracy):
    
    log_data = {
        "test_loss": [test_loss],
        "test_accuracy": [test_accuracy],
    }
    
    log_df = pd.DataFrame(log_data)
    
    save_file = os.path.join(save_folder, filename)

    try:
        log_df.to_csv(save_file, mode='a', header=not pd.io.common.file_exists(filename), index=False)
    except Exception as e:
        print(f"Error logging experiment: {e}")


def log_train_metrics(epochs, train_losses, train_accuracies):
    train_metrics = pd.DataFrame({
            'Epoch': epochs,
            'Train Loss': train_losses,
            'Train Accuracy': train_accuracies
        })
    
    train_save_path = os.path.join(save_folder, train_f_name)
    train_metrics.to_csv(train_save_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy over Epochs for Final Model')
    plt.legend()
    plt.grid(True)

    # Save the plot as a PNG file
    plt.savefig(os.path.join(save_folder, 'training_metrics_plot_final_model.png'), format='png')


def transforming_data(best_params):
    
    train_transforms = transforms.Compose([
        transforms.RandomRotation(
            degrees=best_params['max_rotation'],
            interpolation=PIL.Image.BILINEAR
        ),
        transforms.ElasticTransform(
            alpha=best_params['max_alpha'],
            sigma=5.0
        ),
        ToTensor()
    ])
    
    train = datasets.MNIST(root="data", download=True, train=True,  transform=train_transforms)
    test = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())

    train_dataset = DataLoader(train, 64)
    test_dataset = DataLoader(test, 64)

    return train_dataset, test_dataset

class MNISTClassifier(nn.Module):
    def __init__(self, batch_norm=False, dropout=0.0):
        super().__init__()
        layers = []
        channels = [32, 64, 64]
        input_channels = 1

        for i, out_channels in enumerate(channels):
            layers.append(nn.Conv2d(input_channels, out_channels, kernel_size=3, padding=1))
            input_channels = out_channels
            
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            
            layers.append(nn.ReLU())
                
            layers.append(nn.MaxPool2d(kernel_size=2))

        layers.append(nn.Flatten())
        layers.append(nn.Linear(576, 128))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(128, 64))
        layers.append(nn.Linear(64, 32))
        layers.append(nn.Linear(32, 10))
        
        # Build the model with selected layers
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def train_model(self, epochs, dataset, loss_fn, opt):
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            correct_predictions = 0
            total_samples = 0
            epoch_loss = 0.0

            with tqdm(total=len(dataset), desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
                for batch_data in dataset:
                    x_data, y_data = batch_data
                    x_data, y_data = x_data.to(device), y_data.to(device)
                    yhat = self.model(x_data)
                    loss = loss_fn(yhat, y_data)

                    opt.zero_grad()
                    loss.backward()
                    opt.step()

                    predictions = torch.argmax(yhat, dim=1)
                    correct_predictions += (predictions == y_data).sum().item()
                    total_samples += y_data.size(0)
                    epoch_loss += loss.item()

                    pbar.set_postfix({"loss": loss.item()})
                    pbar.update(1)

            average_loss = epoch_loss / len(dataset)
            accuracy = correct_predictions / total_samples

            # Log progress
            print(f"Epoch {epoch + 1}: Train Accuracy: {accuracy:.4f}, Average Loss: {average_loss:.4f}")
            
            train_accuracies.append(accuracy)
            train_losses.append(loss.item())
        
        return train_losses, train_accuracies

    def evaluate_model(self, test_dataset, loss_fn):
        self.eval()
        correct_predictions = 0
        total_samples = 0
        total_loss = 0.0

        with torch.no_grad():
            for x_data, y_data in test_dataset:
                x_data, y_data = x_data.to(device), y_data.to(device)
                yhat = self.model(x_data)
                loss = loss_fn(yhat, y_data)
                total_loss += loss.item() * y_data.size(0)
                predictions = torch.argmax(yhat, dim=1)
                correct_predictions += (predictions == y_data).sum().item()
                total_samples += y_data.size(0)

        test_average_loss = total_loss / total_samples
        test_accuracy = correct_predictions / total_samples
        return test_average_loss, test_accuracy


if __name__ == "__main__":

    # load in the study
    study = optuna.load_study(
        study_name=hp_study_name,
        storage=stored_hp_exp
    )

    # get the best parameters
    best_params = study.best_params

    # augment the train set and obtain the test set
    train_dataset, test_dataset = transforming_data(best_params)

    # create an instance of the model
    model = MNISTClassifier(batch_norm=True, dropout=0.25).to(device)
    opt = Adam(
        model.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params['weight_decay']
    )
    loss_fn = nn.CrossEntropyLoss()

    # train the model on the train data
    train_losses, train_accuracies = model.train_model(
        epochs=epochs,
        dataset=train_dataset,
        loss_fn=loss_fn,
        opt=opt
    )

    # log the train metrics
    log_train_metrics(
        epochs=list(range(1, epochs + 1)),
        train_losses=train_losses,
        train_accuracies=train_accuracies,
    )

    # save the model
    with open(final_model, 'wb') as f: 
        save(model.state_dict(), f) 

    # load in the model
    with open(final_model, 'rb') as f: 
        model.load_state_dict(load(f, weights_only=True))

    # test the model
    test_loss, test_accuracy = model.evaluate_model(
        test_dataset=test_dataset,
        loss_fn=loss_fn
    )

    # save the model test metrics
    log_test_metrics(test_loss, test_accuracy)



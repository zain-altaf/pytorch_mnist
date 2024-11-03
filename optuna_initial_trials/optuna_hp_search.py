import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import pandas as pd
import optuna
from optuna import TrialPruned
from pathlib import Path
# constants
device = "cuda" if torch.cuda.is_available() else "cpu"
epochs = 5

optuna.logging.set_verbosity(optuna.logging.INFO)


def objective(trial):
    # this will get a learning rate from the range
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    # range chosen based on what's seen in 2 papers
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    # we need to ensure we have a good number of epochs
    epochs_trial = trial.suggest_int("epochs", 10, 15)

    # Sample augmentation hyperparameters
    augmentations = []

    if trial.suggest_categorical("random_rotation", [True, False]):
        max_rotation_trial = trial.suggest_float("max_rotation", 5.0, 15.0)
        augmentations.append(transforms.RandomRotation(degrees=max_rotation_trial))

    if trial.suggest_categorical("elastic_transform", [True, False]):
        max_alpha_trial = trial.suggest_float("max_alpha", 5.0, 8.0)
        augmentations.append(transforms.ElasticTransform(alpha=max_alpha_trial, sigma=5.0))

    augmentations.append(ToTensor())
    training_transform = transforms.Compose(augmentations)

    # Create datasets and data loaders
    train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=training_transform)
    test_dataset = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Define model, optimizer, and loss function
    model = MNISTClassifier(batch_norm=True, dropout=0.25).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    # train the model
    train_losses, train_accuracies = model.train_model(
        epochs=epochs_trial, dataset=train_loader,
        loss_fn=loss_fn, opt=optimizer
    )

    # save train metrics
    train_metrics = pd.DataFrame({
        "Epoch": list(range(1, epochs_trial + 1)),
        "Train Loss": train_losses,
        "Train Accuracy": train_accuracies,
    })
    Path('optuna_trials_train_metrics').mkdir(exist_ok=True)
    train_metrics.to_csv(f"optuna_trials_train_metrics/train_metrics_trial_{trial.number}.csv", index=False)

    # evaluate model
    test_loss, test_accuracy = model.evaluate_model(test_dataset=test_loader, loss_fn=loss_fn)
    print(f"Test Accuracy: {test_accuracy} Test Loss: {test_loss}")

    # stop the trial if it's not improving based on previous runs
    if trial.should_prune():
        raise TrialPruned()

    return test_accuracy


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

            accuracy = correct_predictions / total_samples
            print(f"Epoch {epoch + 1}: Train Accuracy: {accuracy} Train Loss: {loss.item()}")
            
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

    # create a study that we want to maximize
    study = optuna.create_study(
        study_name="hyperparameter_search_mnist",
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        storage="sqlite:///optuna_study.db"
    )
    study.optimize(objective, n_trials=25)


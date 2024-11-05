import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import pandas as pd
import optuna
from pathlib import Path
from classifier_model import MNISTClassifier

# constants
device = "cuda" if torch.cuda.is_available() else "cpu"
save_folder = 'optuna_trial_metrics'

# optuna trial ranges
num_trials = 10
lr_range = [1e-4, 1e-3]
wd_range = [1e-6, 1e-4]
epoch_range = [10, 15]
rotation_range = [5.0, 15.0]
alpha_range = [5.0, 8.0]

# regularization
bn = True
do = 0.25

# this will log important information during training and evaluation
optuna.logging.set_verbosity(optuna.logging.INFO)

def objective(trial):
    # this will get a learning rate from the range
    learning_rate = trial.suggest_float("learning_rate", lr_range[0], lr_range[1], log=True)
    # range chosen based on what's seen in 2 papers
    weight_decay = trial.suggest_float("weight_decay", wd_range[0], wd_range[1], log=True)
    # we need to ensure we have a good number of epochs
    epochs_trial = trial.suggest_int("epochs", epoch_range[0], epoch_range[1])

    # Sample augmentation hyperparameters
    augmentations = []

    max_rotation_trial = trial.suggest_float("max_rotation", rotation_range[0], rotation_range[1])
    augmentations.append(transforms.RandomRotation(degrees=max_rotation_trial))

    max_alpha_trial = trial.suggest_float("max_alpha", alpha_range[0], alpha_range[1])
    augmentations.append(transforms.ElasticTransform(alpha=max_alpha_trial, sigma=5.0))

    augmentations.append(ToTensor())
    training_transform = transforms.Compose(augmentations)

    # Create datasets and data loaders
    train_dataset = datasets.MNIST(root="data", download=True, train=True, transform=training_transform)
    test_dataset = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Define model, optimizer, and loss function
    model = MNISTClassifier(batch_norm=bn, dropout=do).to(device)
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
    Path(save_folder).mkdir(exist_ok=True)
    train_metrics.to_csv(f"{save_folder}/train_metrics_trial_{trial.number}.csv", index=False)

    # evaluate model
    test_loss, test_accuracy = model.evaluate_model(test_dataset=test_loader, loss_fn=loss_fn)
    print(f"Test Accuracy: {test_accuracy} Test Loss: {test_loss}")

    # we are trying to optimize test accuracy
    return test_accuracy


if __name__ == "__main__":

    # create a study that we want to maximize
    study = optuna.create_study(
        study_name="hyperparameter_search_mnist",
        direction='maximize',
        sampler=optuna.samplers.RandomSampler(seed=42),
        storage=f"sqlite:///{save_folder}/optuna_study.db"
    )
    study.optimize(objective, n_trials=num_trials)
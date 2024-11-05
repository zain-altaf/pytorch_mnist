import torch
from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import PIL
import pandas as pd
import matplotlib.pyplot as plt
import os
import optuna
from classifier_model import MNISTClassifier

# Constants
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
epochs = 15
stored_hp_exp = "sqlite:///optuna_trial_metrics/optuna_study.db"
save_folder = "final_model"
final_model = 'final_model.pt'
hp_study_name = "hyperparameter_search_mnist_2"
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

    # create the folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

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



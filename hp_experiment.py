import torch
from torch import nn
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
from classifier_model import MNISTClassifier

# Constants
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_default_device(device)
epochs = 10
max_rotation = 15
max_translation = 0.1
max_alpha = 8
max_kernel = 3
num_experiments = 10
save_folder = "explore_hps"

transformations = [
    "random rotation",
    "random translation",
    "elastic transform",
    "gaussian blur"
]


def log_hyperparameters(hps_for_run, hp_vals_for_run, test_loss, test_accuracy, batch_norm, dropout, exp_num):
    
    hp_vals_for_run = [str(val) for val in hp_vals_for_run]
    
    log_data = {
        "augmentation hyperparameters": [', '.join(hps_for_run)],
        "augmentation hyperparameter values": [', '.join(hp_vals_for_run)],
        "test_loss": [test_loss],
        "test_accuracy": [test_accuracy],
        "batch_norm": [batch_norm],
        "dropout": [dropout]
    }
    
    log_df = pd.DataFrame(log_data)
    
    filename = f"experiment_{exp_num}_hyperparameters.csv"
    save_file = os.path.join(save_folder, filename)

    try:
        log_df.to_csv(save_file, mode='a', header=not pd.io.common.file_exists(filename), index=False)
    except Exception as e:
        print(f"Error logging experiment: {e}")


def log_train_metrics(epochs, train_losses, train_accuracies, exp_num):
    train_metrics = pd.DataFrame({
        'Epoch': epochs,
        'Train Loss': train_losses,
        'Train Accuracy': train_accuracies
    })
    
    train_f_name = f"exp_{exp_num}_train_loss_and_accuracy.csv"
    train_save_path = os.path.join(save_folder, train_f_name)
    train_metrics.to_csv(train_save_path, index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, train_accuracies, label='Train Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Loss and Accuracy over Epochs')
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(save_folder, f'training_metrics_plot_experiment_{exp_num}.png'), format='png')


def get_train_transforms(hps_for_run):
    random_transformations = []
    random_transformation_val = []

    # use (Cubuck et al., 2020) to experiment with
    # augmentation hyperparameters
    variation_from_max = random.uniform(0.65, 0.75)

    # from the randomly chosen augmentations, prepare them
    # for the train set
    for hp in hps_for_run:
        if hp == 'random rotation':
            varied_max_rotation = max_rotation * variation_from_max
            random_transformation_val.append(varied_max_rotation)
            random_rotation = transforms.RandomRotation(
                degrees=varied_max_rotation,
                interpolation=PIL.Image.BILINEAR
            )
            random_transformations.append(random_rotation)
        elif hp == 'random translation':
            varied_max_translation = max_translation * variation_from_max
            random_transformation_val.append(varied_max_translation)
            random_translation = transforms.RandomAffine(
                degrees=0,
                translate=(varied_max_translation, varied_max_translation)
            )
            random_transformations.append(random_translation)
        elif hp == 'elastic transform':
            varied_max_alpha = max_alpha * variation_from_max
            random_transformation_val.append(varied_max_alpha)
            random_elastictransform = transforms.ElasticTransform(
                alpha=varied_max_alpha,
                sigma=5.0
            )
            random_transformations.append(random_elastictransform)
        elif hp == 'gaussian blur':
            varied_max_kernel = random.choice([1, 3])
            random_transformation_val.append(varied_max_kernel)
            random_blur = transforms.GaussianBlur(
                kernel_size=(varied_max_kernel, varied_max_kernel)
            )
            random_transformations.append(random_blur)
    
    random_transformations.append(transforms.ToTensor())
    training_transforms = transforms.Compose(random_transformations)
    return training_transforms, random_transformation_val


def transforming_data():
    # randomly choose 2 or 3,
    # this is important for when we choose
    # augmentation hyperparameters
    num_hyperparams = random.randint(2, 3)
    hps_for_run = []

    for _ in range(num_hyperparams):
        random_hp = random.choice([i for i in range(len(transformations)) if transformations[i] not in hps_for_run])
        hps_for_run.append(transformations[random_hp])

    # get the augmentation hyperparameters that were randomly chosen
    training_transforms, hp_vals_for_run = get_train_transforms(hps_for_run)
    train = datasets.MNIST(root="data", download=True, train=True, transform=training_transforms)
    test = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())

    train_dataset = DataLoader(train, 32)
    test_dataset = DataLoader(test, 32)

    return hps_for_run, hp_vals_for_run, train_dataset, test_dataset


def run_experiment(exp_num):
    batch_norm = random.choice([0, 1])
    dropout = random.choice([0.0, 0.25])

    model = MNISTClassifier(batch_norm, dropout).to(device)
    opt = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    hps_for_run, hp_vals_for_run, train_dataset, test_dataset = transforming_data()
    print(f"Augmentation Hyperparameters For Run: {[hp for hp in hps_for_run]}\
          Augmentation Hyperparameter Values for Run {[hp_val for hp_val in hp_vals_for_run]}")
    print(f"Batch Norm: {batch_norm} dropout: {dropout}")
    train_losses, train_accuracies = model.train_model(epochs, train_dataset, loss_fn, opt)
    test_loss, test_accuracy = model.evaluate_model(test_dataset, loss_fn)

    # create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # save the hyper parameters and the test loss + accuracy
    log_hyperparameters(
        hps_for_run, hp_vals_for_run, test_loss,
        test_accuracy, batch_norm, dropout, exp_num
    )

    # save the train metrics and save a png of the train loss and train accuracy
    log_train_metrics(
        epochs=list(range(1, len(train_losses) + 1)),
        train_losses=train_losses,
        train_accuracies=train_accuracies,
        exp_num=exp_num,
    )


if __name__ == "__main__":
    for i in range(num_experiments):
        print(f"Starting Experiment {i+1}")
        run_experiment(i+1)

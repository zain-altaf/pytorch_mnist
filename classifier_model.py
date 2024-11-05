import torch
from torch import nn
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

class MNISTClassifier(nn.Module):

    # this class creates a CNN architecture
    # which is able to train and evaluate a
    # model

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
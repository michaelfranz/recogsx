import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from datasets import get_data_loaders
from model import GenderClassifier
from utils import save_hyperparam, save_plots, create_run, select_device


# Training function.
def train(model, trainloader, optimizer, criterion):
    model.train()
    print('Training')
    train_running_loss = 0.0
    train_running_correct = 0
    counter = 0
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass.
        outputs = model(image)
        # Calculate the loss.
        loss = criterion(outputs, labels)
        train_running_loss += loss.item()
        # Calculate the accuracy.
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()
        # Backpropagation.
        loss.backward()
        # Update the optimizer parameters.
        optimizer.step()

    # Loss and accuracy for the complete epoch.
    epoch_loss = train_running_loss / counter
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    return epoch_loss, epoch_acc


# Validation function.
def validate(model, testloader, criterion, class_names):
    model.eval()
    print('Validation')
    valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            counter += 1

            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            # Forward pass.
            outputs = model(image)
            # Calculate the loss.
            loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            # Calculate the accuracy.
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    # Loss and accuracy for the complete epoch.
    epoch_loss = valid_running_loss / counter
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc


if __name__ == '__main__':
    # Create the current running directory to save plots and hyperparameters.
    run_dir = create_run()

    # Construct the argument parser.
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=1,
                        help='number of epochs to train our network for')
    parser.add_argument(
        '-lr', '--learning-rate', dest='learning_rate', default=0.001,
        type=float, help='learning rate for the optimizer'
    )
    args = vars(parser.parse_args())

    # Write the hyperparameters to a YAML file.
    save_hyperparam(args, f"../outputs/{run_dir}/hyperparam.yml")

    # Learning parameters.
    lr = args['learning_rate']
    epochs = args['epochs']

    device = select_device()
    print(f"Computation device: {device}\n")

    # Build the model.
    model = GenderClassifier().to(device)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.\n")

    # Optimizer.
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Loss function.
    criterion = nn.CrossEntropyLoss()

    # Get the training and validation data loaders.
    train_loader, valid_loader, dataset_classes = get_data_loaders()

    # Lists to keep track of losses and accuracies.
    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []
    # Start the training.
    for epoch in range(epochs):
        print(f"[INFO]: Epoch {epoch + 1} of {epochs}")
        train_epoch_loss, train_epoch_acc = train(model, train_loader,
                                                  optimizer, criterion)
        valid_epoch_loss, valid_epoch_acc = validate(model, valid_loader,
                                                     criterion, dataset_classes)
        train_loss.append(train_epoch_loss)
        valid_loss.append(valid_epoch_loss)
        train_acc.append(train_epoch_acc)
        valid_acc.append(valid_epoch_acc)
        print(f"Training loss: {train_epoch_loss:.3f}, training acc: {train_epoch_acc:.3f}")
        print(f"Validation loss: {valid_epoch_loss:.3f}, validation acc: {valid_epoch_acc:.3f}")
        print('-' * 50)

    # Save the loss and accuracy plots.
    save_plots(
        train_acc, valid_acc, train_loss, valid_loss,
        f"../outputs/{run_dir}/accuracy.png",
        f"../outputs/{run_dir}/loss.png"
    )
    print('TRAINING COMPLETE')

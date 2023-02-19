import torch
import torch.nn as nn
from sklearn.model_selection import GridSearchCV
from skorch import NeuralNetClassifier

from datasets import get_data_loaders
from model import GenderClassifier
from utils import creat_search_run, save_best_hyperparam, select_device

if __name__ == '__main__':
    # Create hyperparam search folder.
    search_folder = creat_search_run()

    # Learning parameters.
    lr = 0.001
    epochs = 20

    # Check for presence of specialised hardware
    device = select_device()
    print(f"Computation device: {device}\n")

    # Loss function. Required for defining `NeuralNetClassifier`
    criterion = nn.CrossEntropyLoss()

    # Instance of `NeuralNetClassifier` to be passed to `GridSearchCV`
    net = NeuralNetClassifier(
        module=GenderClassifier, max_epochs=epochs,
        optimizer=torch.optim.Adam,
        criterion=criterion,
        lr=lr, verbose=1
    )

    # Get the training and validation data loaders.
    train_loader, valid_loader, dataset_classes = get_data_loaders()

    params = {
        'lr': [0.001, 0.01, 0.005, 0.0005],
        'max_epochs': list(range(5, 30, 5)),
    }

    """
    Define `GridSearchCV`.
    """
    gs = GridSearchCV(
        net, params, refit=False, scoring='accuracy', verbose=1, cv=2
    )

    counter = 0
    # search_batches = 2
    for i, data in enumerate(train_loader):
        print(f'Search batch {i+1}')
        counter += 1
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        outputs = gs.fit(image, labels)
        # GridSearch for `search_batches` number of times.
        # if counter == search_batches:
        #     break

    print('SEARCH COMPLETE')
    print("best score: {:.3f}, best params: {}".format(gs.best_score_, gs.best_params_))
    save_best_hyperparam(gs.best_score_, f"../outputs/{search_folder}/best_param.yml")
    save_best_hyperparam(gs.best_params_, f"../outputs/{search_folder}/best_param.yml")

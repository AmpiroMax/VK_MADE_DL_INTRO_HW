""" Training pipeline and result shower module"""

import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from .dataset import CaptchaDataset
from .model import RCNN
from .train import train

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


DEVICE = "cuda"
PATH = "models/"


def training_pipeline():
    train_dataset = CaptchaDataset()
    test_dataset = CaptchaDataset()

    print("Prepearing train data")
    train_dataset.set_state("train")
    print("Prepearing test data")
    test_dataset.set_state("test")

    model = RCNN()
    model.to(DEVICE)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.3)
    criterion = torch.nn.CTCLoss(reduction="sum", zero_infinity=True, blank=36)
    criterion.to(DEVICE)

    history = train(
        train_files=train_dataset,
        test_files=test_dataset,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        epochs=80,
        batch_size=64
    )

    train_loss_history = history[0]
    test_loss_history = history[1]
    train_cer_history = history[2]
    test_cer_history = history[3]

    best_model = history[6]
    torch.save(best_model, PATH + "model")

    plt.subplot(121)
    plt.title("CTC Loss")
    plt.plot(train_loss_history, label="train")
    plt.plot(test_loss_history, label="test")
    plt.legend()

    plt.subplot(122)
    plt.title("CER")
    plt.plot(train_cer_history, label="train")
    plt.plot(test_cer_history, label="test")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    training_pipeline()

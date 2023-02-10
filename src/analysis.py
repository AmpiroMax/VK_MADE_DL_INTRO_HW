""" Module for misclassified word search """

import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics import CharErrorRate

from .dataset import CaptchaDataset
from .model import RCNN
from .train import eval_epoch

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

DEVICE = "cuda"
PATH = "models/"


def searcher_for_problem_examples():
    model = RCNN()
    model = torch.load(PATH + "model")
    model.to(DEVICE)
    model.eval()

    test_dataset = CaptchaDataset()
    test_dataset.set_state("test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        shuffle=False
    )

    criterion = torch.nn.CTCLoss(reduction="sum", zero_infinity=True, blank=36)

    _, predictions, targets = eval_epoch(model, test_loader, criterion)
    cer = CharErrorRate()

    problem_predictions = []
    problem_targets = []
    images = []

    for predicted_word, target_word in zip(predictions, targets):
        if cer(predicted_word, target_word).numpy() > 0.0:
            problem_predictions += ["".join(predicted_word)]
            problem_targets += ["".join(target_word)]

            problem_img = cv2.imread("data/" + problem_targets[-1] + ".png", 0)
            if problem_img is None:
                problem_img = cv2.imread(
                    "data/" + problem_targets[-1] + ".jpg", 0
                )
            images += [problem_img]

    plot_w = int(len(images)**0.5)
    plot_h = int(len(images) // plot_w)

    _, axis = plt.subplots(plot_h, plot_w, figsize=(plot_w * 5, plot_h * 2))
    for i in range(plot_h):
        for j in range(plot_w):
            axis[i, j].imshow(images[i * plot_w + j], cmap="gray")

            curr_target = problem_targets[i * plot_w + j]
            curr_pred = problem_predictions[i * plot_w + j]
            axis[i, j].set_title(
                f"target:     {curr_target} \nprediction: {curr_pred}",
                horizontalalignment='left'
            )

    plt.show()


if __name__ == "__main__":
    searcher_for_problem_examples()

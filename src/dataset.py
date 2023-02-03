""" Dataset implementation module """

import os
import cv2
import torch
import string
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


CHARS = string.digits + string.ascii_lowercase
CHAR_2_LABEL = {
    CHARS[idx]: idx for idx in range(len(CHARS))
}
LABEL_2_CHAR = {
    idx: CHARS[idx] for idx in range(len(CHARS))
}


def convert_label_to_string(label: np.ndarray):
    answer = []
    for sim_idx in label:
        if sim_idx in LABEL_2_CHAR.keys():
            answer += [LABEL_2_CHAR[sim_idx]]
    return answer


def read_image(img_name: str) -> torch.tensor:
    image = cv2.imread(img_name, 0)
    image = cv2.resize(image, (100, 32))
    image = torch.FloatTensor(image)[..., None]
    image = image.permute(2, 0, 1)
    return image


class CaptchaDataset(Dataset):
    def __init__(self) -> None:
        images = []
        labels = []
        path = "data"
        self.state = "train"

        for img_name in tqdm(os.listdir(path)):
            images += [
                read_image(path + "/" + img_name)
            ]
            labels += [
                img_name[:-4]
            ]

        self.img_train, self.img_test, self.label_train, self.label_test = train_test_split(
            images, labels, test_size=0.2, random_state=42
        )

    def set_state(self, state: str) -> None:
        if state not in ["train", "test"]:
            raise Exception("Wrong state, expected train or test")
        self.state = state

    def _label_2_longtensor(self, label: str) -> torch.LongTensor:
        new_label = [CHAR_2_LABEL[sim] for sim in label]
        new_label = torch.LongTensor(new_label)
        return new_label

    def __len__(self):
        if self.state == "train":
            return len(self.img_train)
        if self.state == "test":
            return len(self.img_test)

        raise Exception("Wrong state, expected train or test")

    def __getitem__(self, idx: int):
        if self.state == "train":
            label = self._label_2_longtensor(self.label_train[idx])
            return self.img_train[idx], label

        if self.state == "test":
            label = self._label_2_longtensor(self.label_test[idx])
            return self.img_test[idx], label

        raise Exception("Wrong state, expected train or test")


if __name__ == "__main__":
    print(CHAR_2_LABEL)
    print(LABEL_2_CHAR)

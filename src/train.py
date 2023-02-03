""" Training and evaluationg functions """

from tqdm.auto import tqdm
import torch

from torchmetrics import CharErrorRate
from torch import nn
from torch.utils.data import Dataset, DataLoader

from dataset import convert_label_to_string

DEVICE = "cuda"


def fit_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: torch.nn.CTCLoss,
    optimizer: torch.optim.Optimizer
):

    model.train()
    running_loss = 0.0
    processed_data_size = 0

    predicted_words = []
    target_words = []

    with tqdm(total=len(train_loader)) as pbar:
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(inputs)
            log_probas = torch.nn.functional.log_softmax(logits, dim=2)

            input_lengths = torch.LongTensor(
                [log_probas.size(0)] * log_probas.size(1)
            )
            target_lengths = torch.LongTensor(
                [5 for _ in range(log_probas.size(1))]
            )

            loss = criterion(log_probas, labels, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            pbar.update(1)

            predicted_batch = torch.argmax(
                log_probas.to("cpu").detach(),
                dim=2
            ).permute(1, 0).numpy()
            target_batch = labels.to("cpu").detach().numpy()

            predicted_words += [
                convert_label_to_string(word)
                for word in predicted_batch
            ]

            target_words += [
                convert_label_to_string(word)
                for word in target_batch
            ]

            running_loss += loss
            processed_data_size += inputs.size(0)

    train_loss = float(running_loss / processed_data_size)
    return train_loss, predicted_words, target_words


def eval_epoch(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: torch.nn.CTCLoss,
):

    model.eval()
    running_loss = 0.0
    processed_data = 0

    predicted_words = []
    target_words = []

    with tqdm(total=len(val_loader)) as pbar:
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.set_grad_enabled(False):
                logits = model(inputs)
                log_probas = torch.nn.functional.log_softmax(logits, dim=2)

                input_lengths = torch.LongTensor(
                    [log_probas.size(0)] * log_probas.size(1)
                )
                target_lengths = torch.LongTensor(
                    [5 for _ in range(log_probas.size(1))]
                )

                loss = criterion(log_probas, labels,
                                 input_lengths, target_lengths)

                predicted_batch = torch.argmax(
                    log_probas.to("cpu").detach(),
                    dim=2
                ).permute(1, 0).numpy()
                target_batch = labels.to("cpu").detach().numpy()

                predicted_words += [
                    convert_label_to_string(word)
                    for word in predicted_batch
                ]

                target_words += [
                    convert_label_to_string(word)
                    for word in target_batch
                ]

                running_loss += loss
                processed_data += inputs.size(0)
            pbar.update(1)

    valid_accuracy = float(running_loss / processed_data)
    return valid_accuracy, predicted_words, target_words


def train(
    train_files: Dataset,
    test_files: Dataset,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.CTCLoss,
    epochs: int,
    batch_size: int
):
    train_loader = DataLoader(
        train_files,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_files,
        batch_size=batch_size,
        shuffle=False
    )

    train_loss_history = []
    test_loss_history = []

    train_cer_history = []
    test_cer_history = []

    train_example_history = []
    test_example_history = []

    train_loss = None
    cer = CharErrorRate()

    for epoch in range(epochs):
        print(f"EPOCH {epoch}/{epochs}: ")
        train_loss,  train_predicted_words, train_target_words = fit_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        test_loss,  test_predicted_words, test_target_words = eval_epoch(
            model,
            test_loader,
            criterion
        )
        train_cer = cer(train_predicted_words, train_target_words).numpy()
        test_cer = cer(test_predicted_words, test_target_words).numpy()

        print("Train loss:  ", train_loss)
        print("Train  CER:  ", train_cer)
        print("Test  loss:  ", test_loss)
        print("Test   CER:  ", test_cer)

        train_loss_history += [train_loss]
        test_loss_history += [test_loss]

        train_cer_history += [train_cer]
        test_cer_history += [test_cer]

        train_example_history += [(
            train_predicted_words[0],
            train_target_words[0]
        )]

        test_example_history += [(
            test_predicted_words[0],
            test_target_words[0]
        )]
        print(test_example_history[-1])

    return (train_loss_history, test_loss_history, train_cer_history, test_cer_history, train_example_history, test_example_history)


if __name__ == "__main__":
    print("Hello")

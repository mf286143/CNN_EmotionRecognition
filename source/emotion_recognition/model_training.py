# train loop, validation loop, loss accuracy tracking
# saves model with best val accuracy score


import torch

from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path

# FUNC TRAIN
# train model ofr one epoch and return total loss, total accuracy
def train_one_epoch(
    model: nn.Module,
    dl_train: DataLoader,
    loss_checker: nn.Module,
    weight_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()

    total_loss: float = 0.0
    correct_predictions: int = 0
    total_samples: int = 0

    #pass image batches and labels to chosen device
    for images, labels in dl_train:
        batch_images = images.to(device)
        batch_labels = labels.to(device)
                                                    #TODO: add docummentation regarding those operations (links at the end of the document) respectfully : (1) (2) (3)
        weight_optimizer.zero_grad()                #clear previos batches gradient

        outputs = model(batch_images)               # calculate outputs for current batch

        loss = loss_checker(outputs, batch_labels)  # calculate loss

        loss.backward()                             # calculate loss within modl params

        weight_optimizer.step()                     # update model params using calced gradients


        # TODO: comment this block better
        batch_size = batch_images.size(0)           # get batch size
        total_loss += loss.item() * batch_size      # convert mean loss to sum

        predictions = outputs.argmax(dim = 1)       # choose class index with highest label score
        correct_predictions += ((predictions == batch_labels)
                                .sum().item())      # sum up correctly guesssed labels

        total_samples += batch_size                 # sum up to total samples

    # calculate avg loss, accuracy
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

# FUNC VAL
# evaluate model
def validate_one_epoch(
    model: nn.Module,
    dl_val: DataLoader,
    loss_checker: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()

    total_loss: float = 0.0
    correct_predictions: int = 0
    total_samples: int = 0

    with torch.no_grad():                               # without gradients
        for images, labels in dl_val:              # calculate val loss
            batch_images = images.to(device)
            batch_labels = labels.to(device)

            outputs = model(batch_images)
            loss = loss_checker(outputs, batch_labels)

            # TODO: comment this block better           # block of code is build in similar  way as previous one
            batch_size = batch_images.size(0)
            total_loss += loss.item() * batch_size

            predictions = outputs.argmax(dim = 1)
            correct_predictions += (
                predictions == batch_labels
            ).sum().item()

            total_samples += batch_size

    # calculate avg loss and accuracy for validation set
    average_loss = total_loss / total_samples
    accuracy = correct_predictions / total_samples

    return average_loss, accuracy

# Run training and validation across multiple epochs and save the best model.
def train_model(
    model: nn.Module,
    dl_train: DataLoader,
    dl_val: DataLoader,
    loss_checker: nn.Module,
    weight_optimizer: torch.optim.Optimizer,
    device: torch.device,
    all_epochs: int,
    checkpoint_path: Path,
    class_names: list[str],
) -> dict:
    history = {
        "train_loss": [],
        "train_accuracy": [],
        "validation_loss": [],
        "validation_accuracy": [],
    }

    best_validation_accuracy = -1.0                 # failsave: negative val, so even if first epoch = 0, it ll still bwe saved, because its an "improvement" over -1

    checkpoint_path.parent.mkdir(parents = True, exist_ok = True)

    # train and validate the model once per epoch
    for epoch in range(all_epochs):
        train_loss, train_accuracy = train_one_epoch(
            model = model,
            dl_train = dl_train,
            loss_checker = loss_checker,
            weight_optimizer = weight_optimizer,
            device = device,
        )

        validation_loss, validation_accuracy = validate_one_epoch(
            model = model,
            dl_val = dl_val,
            loss_checker = loss_checker,
            device = device,
        )

        # store train and val for current epoch
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["validation_loss"].append(validation_loss)
        history["validation_accuracy"].append(validation_accuracy)

        # console dsiplay
        print(
            f"Epoch {epoch + 1}/{all_epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"validation loss: {validation_loss:.4f} | "
            f"validation accuracy: {validation_accuracy:.4f}"
        )

        # save checkpoint if validation accuracy is better than the prev one
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": weight_optimizer.state_dict(),
                    "validation_accuracy": validation_accuracy,
                    "validation_loss": validation_loss,
                    "class_names": class_names,
                },
                checkpoint_path,
            )

            print(f"Saved best model: {checkpoint_path}")

    return history

#TODO: CD of previous one
# (1)    https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html
# (2)    https://docs.pytorch.org/docs/2.14/generated/torch.nn.Module.html
# (#)    https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html
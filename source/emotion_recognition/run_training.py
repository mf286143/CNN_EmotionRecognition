# start trainig: load config values, prep data
# prep model, loss function, weight optimizer (Adam)

import tomllib
from pathlib import Path

import torch
from torch import nn

from emotion_recognition.data import create_dl
from emotion_recognition.device_choice_and_seed import (device_choice, set_seed)
from emotion_recognition.model import emotion_cnn
from emotion_recognition.model_training import train_model

def main() -> None:
    project_path = Path(__file__).resolve().parents[2]

    config_path = project_path / "configs" / "baseline.toml"

    with open(config_path, mode = "rb") as file:
        config = tomllib.load(file)

    # make use of device_choice_and_seed
    seed = config["project"] ["seed"]
    set_seed(seed)
    device = device_choice()
    print(f"used device (cpu / cuda): {device}")

    data_path = project_path / config["data"] ["root"]

    # prep data batches for train i val
    dl_train, dl_val, _, class_names = create_dl(
        data_path = str(data_path),
        batch_size = config["data"] ["batch_size"],
        val_split = config["data"] ["val_split"],
        image_size = config["data"] ["image_size"],
        seed = seed,
    )

    # console display / testing output
    print(f"Classes: {class_names}")
    print(f"Train IMG est size: {len(dl_train.dataset)}")
    print(f"Val IMG set size: {len(dl_val.dataset)}")

    # set nn input points so theyre  == dataset labels/groups
    model = emotion_cnn(
        num_classes = len(class_names),
    )

    model = model.to(device)

    #todo: <-- here
    loss_checker = nn.CrossEntropyLoss()                    # for label/class classification

    weight_optimizer = torch.optim.Adam(                    # todo: <-- here2
        model.parameters(),
        lr = config["training"] ["learning_rate"],
    )

    checkpoint_path = (
        project_path / config["outputs"] ["checkpoint_path"]
    )
    # todo cd <-- end here

    # pass params to training loop
    outcomes = train_model(
        model = model,
        dl_train = dl_train,
        dl_val = dl_val,
        loss_checker = loss_checker,
        weight_optimizer = weight_optimizer,
        device = device,
        all_epochs = config["training"] ["all_epochs"],
        checkpoint_path = checkpoint_path,
        class_names = class_names,
    )

    # console output
    print("Training completed-.")
    print(
        "Best val accuracy: "
        f"{max(outcomes['validation_accuracy']):.2%}"
    )
    print(f"Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    main()
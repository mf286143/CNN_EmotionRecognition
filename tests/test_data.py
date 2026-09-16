#Inspects data loaders by printing dataset sizes, tensor shapes,
#pixel ranges, and overlap between training and validation indices.

import tomllib
from pathlib import Path


from emotion_recognition.data import create_dl


def main():
    project_path = Path(__file__).resolve().parents[1]

    with open(
        project_path / "configs" / "baseline.toml",
        mode="rb",
    ) as file:
        config = tomllib.load(file)

    dl_train, dl_val, dl_test, classes = create_dl(
        data_path=str(project_path / config["data"]["root"]),
        batch_size=config["data"]["batch_size"],
        val_split=config["data"]["val_split"],
        image_size=config["data"]["image_size"],
        seed=config["project"]["seed"],
    )

    print("Klasy:", classes)
    print("Zdjęcia treningowe:", len(dl_train.dataset))
    print("Zdjęcia walidacyjne:", len(dl_val.dataset))
    print("Zdjęcia testowe:", len(dl_test.dataset))

    images, labels = next(iter(dl_train))

    print("Wymiary obrazów:", images.shape)
    print("Wymiary etykiet:", labels.shape)
    print("Zakres pikseli:", images.min().item(), images.max().item())
    print("Etykiety pierwszych 10 zdjęć:", labels[:10].tolist())

    train_indexes = set(dl_train.dataset.indices)
    val_indexes = set(dl_val.dataset.indices)

    print(
        "Wspólne indeksy treningu i walidacji:",
        len(train_indexes & val_indexes),
    )


if __name__ == "__main__":
    main()

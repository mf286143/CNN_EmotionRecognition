# Counts images per emotion class in the FER-2013 training dataset
# before the validation split, then saves and displays a bar chart.

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import tomllib
from torchvision import datasets


def main():
    project_path = Path(__file__).resolve().parents[1]

    with open(
        project_path / "configs" / "baseline.toml",
        mode="rb",
    ) as file:
        config = tomllib.load(file)

    train_location = project_path / config["data"]["root"] / "train"
    dataset = datasets.ImageFolder(train_location)

    # Count image files assigned to each class before the validation split.
    class_names = dataset.classes
    class_counts = Counter(label for _, label in dataset.samples)

    print("Rozkład klas w zbiorze treningowym FER-2013:")
    print("Przed wydzieleniem danych walidacyjnych.")
    print()

    for class_index, class_name in enumerate(class_names):
        print(f"{class_name:10s}: {class_counts[class_index]}")

    print(f"Łączna liczba obrazów: {len(dataset)}")

    # Display the number of images in each class as a bar chart.
    counts = [class_counts[index] for index in range(len(class_names))]

    plt.figure(figsize=(10, 6))
    positions = list(range(len(class_names)))
    plt.bar(positions, counts)

    # Show each class count below its bar, beneath the class name.
    tick_labels = [
        f"{class_name}\n{count}"
        for class_name, count in zip(class_names, counts)
    ]
    plt.xticks(positions, tick_labels)

    # Center the total above the plot, below the main title.
    plt.title(
        "Rozkład klas w zbiorze treningowym FER-2013\n"
        f"Łączna liczba obrazów: {sum(counts)}"
    )
    plt.xlabel("Klasa")
    plt.ylabel("Liczba obrazów")
    plt.tight_layout()
    # Save the chart before displaying it.
    output_directory = project_path / "outputs" / "dataset"
    output_directory.mkdir(parents=True, exist_ok=True)
    chart_path = output_directory / "fer2013_dataset_breakdown.png"
    plt.savefig(chart_path, dpi=200, bbox_inches="tight")
    print(f"Wykres zapisano w: {chart_path}")

    plt.show()


if __name__ == "__main__":
    main()

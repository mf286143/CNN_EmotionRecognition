# Provides functions to collect predictions, calculate classification
# metrics, and save and display a confusion matrix.

from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader


def collect_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Zbiera prawdziwe etykiety i predykcje bez aktualizacji wag."""
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for images, labels in data_loader:
            scores = model(images.to(device))
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(scores.argmax(dim=1).cpu().tolist())

    return y_true, y_pred


def calculate_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
) -> tuple[dict, torch.Tensor]:
    """Oblicza metryki; przy zerowym mianowniku przyjmuje wynik 0."""
    if not y_true or len(y_true) != len(y_pred):
        raise ValueError("Etykiety i predykcje muszą być niepuste i tej samej długości.")
    count = len(class_names)
    if count == 0 or len(set(class_names)) != count:
        raise ValueError("Nazwy klas muszą być niepuste i unikalne.")

    true = torch.tensor(y_true, dtype=torch.long)
    predicted = torch.tensor(y_pred, dtype=torch.long)
    if ((true < 0) | (true >= count) | (predicted < 0) | (predicted >= count)).any():
        raise ValueError("Indeks klasy wykracza poza listę nazw klas.")

    # Wiersze oznaczają etykiety prawdziwe, kolumny — przewidywane.
    matrix = torch.bincount(
        true * count + predicted, minlength=count * count,
    ).reshape(count, count)
    true_positive = matrix.diag().double()
    support = matrix.sum(dim=1).double()
    predicted_count = matrix.sum(dim=0).double()
    precision = true_positive / predicted_count.clamp_min(1)
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * true_positive / (support + predicted_count).clamp_min(1)
    total = len(y_true)

    report = {
        name: {
            "precision": precision[index].item(),
            "recall": recall[index].item(),
            "f1-score": f1[index].item(),
            "support": int(support[index].item()),
        }
        for index, name in enumerate(class_names)
    }
    results = {
        "sample_count": total,
        "class_names": class_names,
        "accuracy": true_positive.sum().item() / total,
        "macro_precision": precision.mean().item(),
        "macro_recall": recall.mean().item(),
        "macro_f1": f1.mean().item(),
        "weighted_f1": (f1 * support).sum().item() / total,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }
    return results, matrix


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    confusion_matrix_path: Path,
) -> dict:
    """Ocenia model na całym loaderze i zapisuje macierz pomyłek."""
    y_true, y_pred = collect_predictions(model, data_loader, device)
    results, matrix = calculate_metrics(y_true, y_pred, class_names)

    # Utwórz wykres do zapisania i wyświetlenia w oknie.
    figure = plt.figure(figsize=(10, 8))
    axis = figure.subplots()
    plot = axis.imshow(matrix.numpy(), cmap="Blues")
    figure.colorbar(plot, ax=axis)
    positions = list(range(len(class_names)))
    axis.set_xticks(positions, labels=class_names, rotation=45, ha="right")
    axis.set_yticks(positions, labels=class_names)
    axis.set_xlabel("Przewidywana klasa")
    axis.set_ylabel("Prawdziwa klasa")
    axis.set_title("Macierz pomyłek — zbiór testowy")
    threshold = matrix.max().item() / 2
    for row in positions:
        for column in positions:
            value = matrix[row, column].item()
            axis.text(column, row, str(value), ha="center", va="center",
                      color="white" if value > threshold else "black")
    figure.tight_layout()
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(confusion_matrix_path, dpi=200)

    # Zapisz obraz przed wyświetleniem; zamknij okno, aby kontynuować raport.
    plt.show()
    plt.close(figure)
    return results
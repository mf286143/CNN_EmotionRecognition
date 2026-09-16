# Evaluates the saved model on the test dataset and exports
# classification metrics to JSON and a confusion matrix to PNG.

import json
import tomllib
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from emotion_recognition.data import create_tf
from emotion_recognition.device_choice_and_seed import device_choice
from emotion_recognition.model_prediction import load_trained_model

# Import działa przy uruchomieniu pliku oraz przy zbieraniu plików przez pytest.
if __package__:
    from .test_model_evaluation import evaluate_model
else:
    from test_model_evaluation import evaluate_model


def main() -> None:
    project_path = Path(__file__).resolve().parents[1]
    with open(project_path / "configs" / "baseline.toml", mode="rb") as file:
        config = tomllib.load(file)

    device = device_choice()
    checkpoint_path = project_path / config["outputs"]["checkpoint_path"]
    model, class_names = load_trained_model(checkpoint_path, device)

    # Do końcowej oceny używamy tylko test, bez augmentacji i podziału train/val.
    _, tf_eval = create_tf(config["data"]["image_size"])
    dataset = ImageFolder(
        project_path / config["data"]["root"] / "test",
        transform=tf_eval,
    )
    if dataset.classes != class_names:
        raise ValueError(
            f"Niezgodna kolejność lub nazwy klas. "
            f"Checkpoint: {class_names}; dataset: {dataset.classes}"
        )
    data_loader = DataLoader(
        dataset, batch_size=config["data"]["batch_size"],
        shuffle=False, num_workers=0,
    )

    # Ścieżki wyników odpowiadają obecnej strukturze katalogu outputs.
    output_path = project_path / "outputs" / "baseline_model"
    matrix_path = output_path / "plots" / "test_confusion_matrix.png"
    report_path = output_path / "evaluation_reports" / "test_metrics.json"

    print(f"Urządzenie: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Liczba zdjęć testowych: {len(dataset)}", flush=True)
    results = evaluate_model(model, data_loader, class_names, device, matrix_path)
    results["checkpoint_path"] = str(checkpoint_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, mode="w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4, allow_nan=False)

    print(f"\nAccuracy: {results['accuracy']:.2%}")
    print(f"Macro precision: {results['macro_precision']:.4f}")
    print(f"Macro recall: {results['macro_recall']:.4f}")
    print(f"Macro F1: {results['macro_f1']:.4f}")
    print(f"Weighted F1: {results['weighted_f1']:.4f}")
    print(f"\n{'Klasa':12s} {'Precision':>10s} {'Recall':>10s} {'F1':>10s} {'Liczba':>8s}")
    for name, metrics in results["classification_report"].items():
        print(f"{name:12s} {metrics['precision']:10.4f} {metrics['recall']:10.4f} "
              f"{metrics['f1-score']:10.4f} {metrics['support']:8d}")
    print(f"\nRaport: {report_path}")
    print(f"Macierz pomyłek: {matrix_path}")


if __name__ == "__main__":
    main()
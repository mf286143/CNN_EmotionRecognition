# Checks the model using random input images, verifies output shape
# and finite values, and prints the architecture and parameter count.

from pathlib import Path

import torch
import tomllib

from emotion_recognition.model import emotion_cnn


def main():
    project_path = Path(__file__).resolve().parents[1]
    with open(
        project_path / "configs" / "baseline.toml",
        mode="rb",
    ) as file:
        config = tomllib.load(file)

    batch_size = config["data"]["batch_size"]
    image_size = config["data"]["image_size"]
    num_classes = 7

    torch.manual_seed(config["project"]["seed"])
    model = emotion_cnn(num_classes=num_classes)
    model.eval()

    print("Architektura modelu:")
    print(model)

    # Check the forward pass using synthetic grayscale images.
    images = torch.rand(batch_size, 1, image_size, image_size)
    with torch.no_grad():
        outputs = model(images)

    print("Wymiary wejścia:", images.shape)
    print("Wymiary wyjścia:", outputs.shape)
    print("Liczba parametrów:", sum(p.numel() for p in model.parameters()))

    expected_shape = (batch_size, num_classes)
    assert tuple(outputs.shape) == expected_shape, (
        f"Niepoprawny kształt wyjścia: {tuple(outputs.shape)}; "
        f"oczekiwano: {expected_shape}"
    )
    assert torch.isfinite(outputs).all().item(), (
        "Wyjście modelu zawiera NaN lub nieskończoność."
    )

    print("Test zakończony poprawnie.")


if __name__ == "__main__":
    main()

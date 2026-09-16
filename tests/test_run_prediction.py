# Runs predictions on example images and an optional user-provided image,
# displaying predicted emotions, class probabilities, and execution time.

import time
import torch
import tomllib
from pathlib import Path

from emotion_recognition.device_choice_and_seed import device_choice
from emotion_recognition.model_prediction import (
    load_trained_model,
    preprocess_image,
    predict_image,
)


def show_prediction(
    image_path,
    model,
    class_names,
    device,
    image_size,
) -> None:
    """Wykonuje predykcję dla zdjęcia i wyświetla wyniki."""
    image_tensor = preprocess_image(
        image_path=image_path,
        image_size=image_size,
    )

    # Poczekaj na zakończenie wcześniejszych operacji GPU.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()

    result = predict_image(
        model=model,
        image_tensor=image_tensor,
        class_names=class_names,
        device=device,
    )

    # GPU działa asynchronicznie — zaczekaj na wynik przed końcem pomiaru.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed_time = time.perf_counter() - start_time

    print(f"\nZdjęcie: {image_path}")
    print(f"Przewidywana emocja: {result['predicted_class']}")
    print(
        "Prawdopodobieństwo wybranej klasy: "
        f"{result['confidence']:.2%}"
    )

    sorted_probabilities = sorted(
        result["probabilities"].items(),
        key=lambda item: item[1],
        reverse=True,
    )

    print("Prawdopodobieństwa klas:")

    for class_name, probability in sorted_probabilities:
        print(f"  {class_name:12s}: {probability:.2%}")

    print(
        f"Czas predykcji: {elapsed_time:.4f} s "
        f"({elapsed_time * 1000:.2f} ms)"
    )


def main() -> None:
    project_path = Path(__file__).resolve().parents[1]
    config_path = project_path / "configs" / "baseline.toml"

    with open(config_path, mode="rb") as file:
        config = tomllib.load(file)

    device = device_choice()
    print(f"Urządzenie obliczeniowe: {device}")

    checkpoint_path = (
        project_path / config["outputs"]["checkpoint_path"]
    )

    # Wczytaj model raz dla wszystkich zdjęć.
    model, class_names = load_trained_model(
        checkpoint_path=checkpoint_path,
        device=device,
    )

    image_size = config["data"]["image_size"]

    # Kolejność listy określa kolejność wykonywania predykcji.
    test_data_path = project_path / "tests" / "data_for_testing"

    test_images = [
        test_data_path / "happy_full_body.jpg",
        test_data_path / "happy_cropped_face.jpg",
    ]

    for image_path in test_images:
        show_prediction(
            image_path=image_path,
            model=model,
            class_names=class_names,
            device=device,
            image_size=image_size,
        )

    # Dwa zdjęcia testowe zostały już sprawdzone.
    # Enter kończy program, a podanie ścieżki dodaje trzecią predykcję.
    entered_path = input(
        "\nPodaj ścieżkę dodatkowego zdjęcia "
        "\nformat przykładowej ścieżki ( do do folderu w którym znajduje się jednego z wyżej testowanych zdjęć):"
        "\n\ntests/data_for_testing/happy_cropped_face.jpg\n"
        "\nlub naciśnij Enter, aby zakończyć: "
    ).strip()

    if not entered_path:
        return

    # Pozwól wkleić ścieżkę otoczoną cudzysłowami.
    entered_path = entered_path.strip("\"'")

    if not entered_path:
        print("Podano niepoprawną ścieżkę.")
        return

    image_path = Path(entered_path).expanduser()

    # Ścieżkę względną odczytuj względem katalogu projektu.
    if not image_path.is_absolute():
        image_path = project_path / image_path

    if not image_path.is_file():
        print("Podano niepoprawną ścieżkę.")
        return

    # Plik może istnieć, ale nie być poprawnym obrazem.
    try:
        show_prediction(
            image_path=image_path,
            model=model,
            class_names=class_names,
            device=device,
            image_size=image_size,
        )
    except (OSError, ValueError) as error:
        print(f"Nie udało się odczytać lub przetworzyć zdjęcia: {error}")


if __name__ == "__main__":
    main()
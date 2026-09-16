# load mode, preprocess input image, predict emotion class/label

from pathlib import Path
from PIL import Image

import torch

from emotion_recognition.model import emotion_cnn
from emotion_recognition.data import create_tf

# load stored weights, prepare predict model on selected device (cpu gpu) and give output including class names/labels
def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[emotion_cnn, list[str]]:
    #"""delete this later"""         #todo: delete this later

    checkpoint = torch.load(
        checkpoint_path,
        map_location = device,
        weights_only = True,
    )

    class_names = checkpoint["class_names"]             # get classnames/labels

    # make nn with input points == class names/labels
    model = emotion_cnn(
        num_classes=len(class_names),
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)                            # use selected device

    model.eval()                                        # evaluate mode

    return model, class_names

def preprocess_image(
    image_path: Path,
    image_size: int,
) -> torch.Tensor:
    #""""delete"""                                       #todo: delete later

    # gets img transformators without augmentaction tilt, flip
    _, tf_eval = create_tf(image_size)

    # open img and preprocess same as validation set
    with Image.open(image_path) as image:
        image_tensor = tf_eval(image.convert("RGB"))

    image_tensor = image_tensor.unsqueeze(0)            # adds batchsize on the front of tensor (?)

    return image_tensor

def predict_image(
    model: emotion_cnn,
    image_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
) -> dict:
    """"delete"""               #todo <-------- delete

    # use selected device
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        class_score = model(image_tensor)

        # calculate class/label probability out of data
        probabilities = torch.softmax(class_score, dim = 1)[0]

        # sort labels descending
        predicted_index = int(probabilities.argmax().item())        # added int()

        # set class scrore prediction to correct class
        predicted_class = class_names[predicted_index]
        confidence = probabilities[predicted_index].item()

    # combine all classes + probabilities
    class_probabilities = dict(
        zip(
            class_names,
            probabilities.cpu().tolist(),
        )
    )


    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": class_probabilities,
    }

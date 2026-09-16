# select device tha twill be used for computation
# set seed

import torch


# selects CUDA if available for faster nn learning
def device_choice() -> torch.device:
    if torch.cuda.is_available():
        chosen_device = torch.device("cuda")
    else:
        chosen_device = torch.device("cpu")

    return chosen_device

# set seed to make experiments reproductable
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
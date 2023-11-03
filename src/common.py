import torch


def get_device() -> torch.device:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = 'mps'

    return torch.device(device)

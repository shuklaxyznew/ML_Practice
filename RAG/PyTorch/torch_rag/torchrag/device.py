import torch


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("[Device] Apple Silicon (MPS)")
        return torch.device("mps")
    print("[Device] CPU")
    return torch.device("cpu")

from typing import List
import torch

def get_device_name(device_name: str) -> str:
    if device_name is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if device_name == "cpu":
        return "cpu"
    if device_name.startswith("cuda:"):
        return device_name
    if device_name.isnumeric():
        return f"cuda:{device_name}"
    else:
        raise RuntimeError(
            f"Device name: {device_name} not recognized. "
            f"Either do not set, set to cpu, or give a number"
        )

def get_device_names(device_name_str: str) -> List[str]:
    if device_name_str is None or "," not in device_name_str:
        return [get_device_name(device_name_str)]
    else:
        return [get_device_name(x.strip()) for x in device_name_str.split(",") if len(x.strip()) > 0]


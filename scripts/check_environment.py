"""Print the installed framework and available training device."""
from importlib.metadata import version

import torch

print(f"nnU-Net: {version('nnunetv2')}")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {props.name}, {props.total_memory / 1024**3:.1f} GiB VRAM")
    # Check an actual operation, not just driver discovery.
    probe = torch.ones((16, 16), device='cuda')
    assert (probe @ probe)[0, 0].item() == 16
    print('CUDA tensor operation: OK')
else:
    print('CPU available; full nnU-Net training may be slow.')

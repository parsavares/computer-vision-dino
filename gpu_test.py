import torch

print("torch version:", torch.__version__)
print("cuda available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("cuda device count:", torch.cuda.device_count())
    print("current device index:", torch.cuda.current_device())
    print("current device name:", torch.cuda.get_device_name(torch.cuda.current_device()))


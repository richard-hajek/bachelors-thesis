import torch
import torchvision

print("Torch Version, Torchvision Version")
print(torch.__version__)
print(torchvision.__version__)


print("Torch with CUDA?")
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

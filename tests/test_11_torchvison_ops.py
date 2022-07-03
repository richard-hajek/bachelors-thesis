import torch
import torchvision
from torchvision import ops
import unittest

class TorchvisionOPS(unittest.TestCase):

    def test_ops(self):
        print("torch: ", torch.__file__)
        print("torchvision: ", torchvision.__file__)
        ops.nms(torch.Tensor([[1, 2, 3, 4]]).to("cuda"), torch.Tensor([1]).to("cuda"), iou_threshold=0.5)
        print("ops OK")

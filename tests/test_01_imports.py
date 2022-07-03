import unittest


class TestStringMethods(unittest.TestCase):

    def test_versions(self):
        import torch
        import torchvision
        print("Torch Version, Torchvision Version")
        print(torch.__version__)
        print(torchvision.__version__)

    def test_cuda(self):
        import torch
        import torchvision

        self.assertTrue(torch.cuda.is_available())
        assert torch.cuda.device_count() > 0
        assert torch.cuda.current_device() == 0
        assert torch.cuda.get_device_name(0)


if __name__ == '__main__':
    unittest.main()

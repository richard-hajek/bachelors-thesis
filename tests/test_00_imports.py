import importlib
import unittest


class TestImports(unittest.TestCase):
    def test_imports(self):
        for name in ["satnet", "torch"]:
            try:
                importlib.import_module(name)
                print(f"Module {name} available")
            except Exception as e:
                print(f"Failed to import {name} due to {e}")

    def test_env(self):
        import os
        assert os.environ.get("CUDA_HOME", "") != "", f"CUDA_HOME is {os.environ.get('CUDA_HOME')}"
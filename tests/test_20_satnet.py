from unittest import TestCase

import satnet
from torch import nn

print('SATNet document\n', satnet.SATNet.__doc__)

class SudokuSolver(nn.Module):
    def __init__(self, boardSz, aux, m):
        super(SudokuSolver, self).__init__()
        n = boardSz**6
        self.sat = satnet.SATNet(n, m, aux)

    def forward(self, y_in, mask):
        out = self.sat(y_in, mask)
        del y_in, mask
        return out


class TestSATNet(TestCase):

    def test_satnet(self):
        solver = SudokuSolver(5, 5, 5)


if __name__ == "__main__":
    pass
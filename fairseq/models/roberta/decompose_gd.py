"""Decompose RoBERTa pretrained model into low-rank matrices."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Parameter

__author__ = 'v-yaf'


class MatrixDecomposer(nn.Module):
    def __init__(self, mat, k):
        super().__init__()
        assert mat.ndimension() == 2, 'input matrix must be 2-dim'
        m, n = mat.size()
        self.mat_a = Parameter(mat.new(m, k))
        self.mat_b = Parameter(mat.new(k, n))

        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.mat_a)
        nn.init.xavier_uniform_(self.mat_b)
    
    def forward(self):
        return torch.mm(self.mat_a, self.mat_b)


def _loss(mat, f_out):
    return torch.mean((mat - f_out) ** 2)


def decompose_matrix(mat, k, patience=20):
    decomposer = MatrixDecomposer(mat, k)
    # optimizer = optim.Adam(decomposer.parameters())
    optimizer = optim.SGD(decomposer.parameters(), lr=0.1)

    step = 0
    counter = 0
    best_loss = 1e20
    best_decomposer = MatrixDecomposer(mat, k)
    current_loss = best_loss - 1

    while True:
        step += 1
        if current_loss >= best_loss:
            counter += 1
            if counter >= patience:
                break
        else:
            best_loss = current_loss
            best_decomposer.load_state_dict(decomposer.state_dict())

        output = decomposer()
        loss = _loss(mat, output)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        print('|', current_loss)

    return best_decomposer.mat_a.data, best_decomposer.mat_b.data


def main():
    import numpy as np
    x = torch.from_numpy(np.random.rand(1024, 1024))
    k = 1024

    a, b = decompose_matrix(x, k)

    print('Final:', _loss(x, torch.mm(a, b)).item())
    print(torch.mm(a, b)[:5, :5])
    print(x[:5, :5])


if __name__ == "__main__":
    main()

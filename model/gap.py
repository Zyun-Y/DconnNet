import torch
import torch.nn as nn

GlobalAvgPool2D = lambda: nn.AdaptiveAvgPool2d(1)


class GlobalAvgPool2DBaseline(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2DBaseline, self).__init__()

    def forward(self, x):
        x_pool = torch.mean(x.view(x.size(0), x.size(1), x.size(2) * x.size(3)), dim=2)

        x_pool = x_pool.view(x.size(0), x.size(1), 1, 1).contiguous()
        return x_pool

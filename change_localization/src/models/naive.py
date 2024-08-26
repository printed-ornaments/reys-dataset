import torch.nn as nn
import torch


class Naive(nn.Module):
    def __init__(
            self,
            mean,
            samples
    ):
        super(Naive, self).__init__()
        self.samples = samples
        if mean is None:
            mean = torch.mean(samples, dim=0)
        self.mean = mean
        self.num_samples = samples.size(0)

    def forward(self):
        mean = self.mean.unsqueeze(0).expand(self.num_samples, -1, -1, -1)
        return ((mean - self.samples)**2).mean(), mean

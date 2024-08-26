import torch
import torch.nn as nn
import torch.nn.functional as F


class Congealing(nn.Module):
    def __init__(
            self,
            mask,
            samples,
            learn_prototype
    ):
        super(Congealing, self).__init__()
        self.padding_mode = 'border'
        self.color_ch = 3
        self.samples = samples
        if mask is None:
            mask = torch.mean(samples, dim=0)
            mask = (mask.mean(0) < 0.5).float() * 2 - 1
        self.mask = nn.Parameter(mask, requires_grad=learn_prototype)
        self.num_samples = samples.size(0)
        self.mask_color = nn.Parameter(torch.zeros((self.num_samples, 3)))
        self.background_color = nn.Parameter(torch.ones((self.num_samples, 3)))
        self.affine_params = nn.Parameter(torch.zeros((self.num_samples, 2, 3)))
        self.register_buffer('affine_identity', torch.cat([torch.eye(2, 2), torch.zeros(2, 1)], dim=1))

    def _affine_transform(self, x, inverse=False):
        beta = self.affine_params.view(-1, 2, 3) + self.affine_identity
        if inverse:
            row = torch.tensor([[[0, 0, 1]]] * x.size(0), dtype=torch.float, device=beta.device)
            beta = torch.cat([beta, row], dim=1)
            beta = torch.inverse(beta)[:, :2, :]
        grid = F.affine_grid(beta, x.size(), align_corners=False)
        return F.grid_sample(x, grid, padding_mode=self.padding_mode, align_corners=False)

    def forward(self):
        mask = torch.sigmoid(self.mask.unsqueeze(0).expand(self.num_samples, -1, -1, -1))
        background = self.background_color[..., None, None].expand(-1, -1, mask.size(2), mask.size(3))
        recons_mask = self._affine_transform(mask)
        recons = recons_mask * self.mask_color[..., None, None] + (1-recons_mask) * background
        return ((recons - self.samples)**2).mean(), recons

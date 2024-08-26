import torch
import torch.nn as nn
import torch.nn.functional as F


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims, feature_size, height, width):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(height * width * 3, feature_size)
        self.linear2 = nn.Linear(feature_size, latent_dims)
        self.linear3 = nn.Linear(feature_size, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x, mode='train'):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        if mode == 'train':
            z = mu + sigma * self.N.sample(mu.shape)
        else:
            z = mu
        self.kl = (sigma ** 2 + mu ** 2 - torch.log(sigma) - 1 / 2).sum()
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dims, feature_size, height, width):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, feature_size)
        self.linear2 = nn.Linear(feature_size, height*width*3)
        self.height, self.width = height, width

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 3, self.height, self.width))


class VariationalAutoencoder(nn.Module):
    def __init__(self, samples, latent_dims, feature_size, height, width):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims, feature_size, height, width)
        self.decoder = Decoder(latent_dims, feature_size, height, width)
        self.samples = samples
        self.gt_samples = samples

    def forward(self, mode='train'):
        z = self.encoder(self.samples, mode=mode)
        recons = self.decoder(z)
        return((recons - self.gt_samples)**2).sum(), recons

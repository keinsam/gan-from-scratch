import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,
                 channel_dim: int = 3,
                 hidden_dim: int = 64
                ) -> None:
        super().__init__()
        self.channel_dim = channel_dim
        self.hidden_dim = hidden_dim

        self.block1 = self._block(channel_dim, hidden_dim // 4, kernel_size=3, stride=2, padding=1)
        self.block2 = self._block(hidden_dim // 4, hidden_dim // 2, kernel_size=3, stride=2, padding=1)
        self.block3 = self._block(hidden_dim // 2, hidden_dim, kernel_size=3, stride=2, padding=1)
        self.flatten = nn.Flatten()
        self.conv_to_output = nn.Sequential(nn.Linear(hidden_dim * 4 * 4, 1),
                                            nn.Sigmoid()) # nn.Tanh()

    def _block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.conv_to_output(x)
        return x


class Generator(nn.Module):
    def __init__(self, channel_dim: int = 3, hidden_dim: int = 64, latent_dim: int = 128):
        super().__init__()
        self.channel_dim = channel_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.latent_to_conv = nn.Linear(latent_dim, hidden_dim * 4 * 4)
        self.block1 = self._block(hidden_dim, hidden_dim // 2, kernel_size=3, stride=2, padding=1)
        self.block2 = self._block(hidden_dim // 2, hidden_dim // 4, kernel_size=3, stride=2, padding=1)
        self.block3 = self._block(hidden_dim // 4, channel_dim, kernel_size=3, stride=2, padding=1)

    def _block(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
        )
    
    def initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)

    def forward(self, z):
        x = self.latent_to_conv(z)
        x = x.view(-1, self.hidden_dim, 4, 4)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
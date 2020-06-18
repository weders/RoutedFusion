import torch

from torch import nn


class FusionNet(nn.Module):

    def __init__(self, config):

        super(FusionNet, self).__init__()

        self.uncertainty = config.uncertainty

        self.n_channels = 2 * config.n_points + 1 + int(config.confidence)
        self.n_points = config.n_points

        self.block1 = nn.Sequential(nn.Conv2d(self.n_channels, self.n_channels, (3, 3), padding=1),
                                    nn.BatchNorm2d(self.n_channels),
                                    nn.LeakyReLU(),
                                    nn.Dropout2d(p=0.2),
                                    nn.Conv2d(self.n_channels, self.n_channels, (3, 3), padding=1),
                                    nn.BatchNorm2d(self.n_channels),
                                    nn.LeakyReLU(),
                                    nn.Dropout2d(p=0.2))

        self.block2 = nn.Sequential(nn.Conv2d(2 * self.n_channels, self.n_channels, (3, 3), padding=1),
                                    nn.BatchNorm2d(self.n_channels),
                                    nn.LeakyReLU(),
                                    nn.Dropout2d(p=0.2),
                                    nn.Conv2d(self.n_channels, self.n_channels, (3, 3), padding=1),
                                    nn.BatchNorm2d(self.n_channels),
                                    nn.LeakyReLU(),
                                    nn.Dropout2d(p=0.2))

        self.block3 = nn.Sequential(nn.Conv2d(3 * self.n_channels, self.n_channels, (3, 3), padding=1),
                                    nn.BatchNorm2d(self.n_channels),
                                    nn.LeakyReLU(),
                                    nn.Dropout2d(p=0.2),
                                    nn.Conv2d(self.n_channels, self.n_channels, (3, 3), padding=1),
                                    nn.BatchNorm2d(self.n_channels),
                                    nn.LeakyReLU(),
                                    nn.Dropout2d(p=0.2))

        self.block4 = nn.Sequential(nn.Conv2d(4 * self.n_channels, self.n_channels, (3, 3), padding=1),
                                    nn.BatchNorm2d(self.n_channels),
                                    nn.LeakyReLU(),
                                    nn.Dropout2d(p=0.2),
                                    nn.Conv2d(self.n_channels, self.n_channels, (3, 3), padding=1),
                                    nn.BatchNorm2d(self.n_channels),
                                    nn.LeakyReLU(),
                                    nn.Dropout2d(p=0.2))

        self.pred1 = nn.Sequential(nn.Conv2d(5 * self.n_channels, 4 * self.n_channels, (1, 1), padding=0),
                                   nn.BatchNorm2d(4 * self.n_channels),
                                   nn.LeakyReLU(),
                                   nn.Dropout2d(p=0.2),
                                   nn.Conv2d(4 * self.n_channels, 4 * self.n_channels, (1, 1), padding=0),
                                   nn.BatchNorm2d(4 * self.n_channels),
                                   nn.LeakyReLU(),
                                   nn.Dropout2d(p=0.2))

        self.pred2 = nn.Sequential(nn.Conv2d(4 * self.n_channels, 3 * self.n_channels, (1, 1), padding=0),
                                   nn.BatchNorm2d(3 *self.n_channels),
                                   nn.LeakyReLU(),
                                   nn.Dropout2d(p=0.2),
                                   nn.Conv2d(3 * self.n_channels, 3 * self.n_channels, (1, 1), padding=0),
                                   nn.BatchNorm2d(3 * self.n_channels),
                                   nn.LeakyReLU(),
                                   nn.Dropout2d(p=0.2))

        self.pred3 = nn.Sequential(nn.Conv2d(3 * self.n_channels, 2 * self.n_channels, (1, 1), padding=0),
                                   nn.BatchNorm2d(2 * self.n_channels),
                                   nn.LeakyReLU(),
                                   nn.Dropout2d(p=0.2),
                                   nn.Conv2d(2 * self.n_channels, 2 * self.n_channels, (1, 1), padding=0),
                                   nn.BatchNorm2d(2 * self.n_channels),
                                   nn.LeakyReLU(),
                                   nn.Dropout2d(p=0.2))

        self.pred4 = nn.Sequential(nn.Conv2d(2 * self.n_channels, 1 * self.n_channels, (1, 1), padding=0),
                                   nn.BatchNorm2d(self.n_channels),
                                   nn.LeakyReLU(),
                                   nn.Dropout2d(p=0.2),
                                   nn.Conv2d(1 * self.n_channels, 1 * self.n_channels, (1, 1), padding=0),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(1 * self.n_channels, self.n_points, (1, 1), padding=0),
                                   nn.Tanh())

    def forward(self, x):

        x1 = self.block1.forward(x)
        x1 = torch.cat([x, x1], dim=1)
        x2 = self.block2.forward(x1)
        x2 = torch.cat([x1, x2], dim=1)
        x3 = self.block3.forward(x2)
        x3 = torch.cat([x2, x3], dim=1)
        x4 = self.block4.forward(x3)
        x4 = torch.cat([x3, x4], dim=1)

        y = self.pred1.forward(x4)
        y = self.pred2.forward(y)
        y = self.pred3.forward(y)
        y = self.pred4.forward(y)

        del x1, x2, x3, x4

        return y

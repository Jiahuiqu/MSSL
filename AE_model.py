import torch
import torch.nn as nn


class AE3D(nn.Module):

    def __init__(self):

        super(AE3D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv3d(1,4,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(True),
            nn.Conv3d(4,2,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(2),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(2,4,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(4),
            nn.ReLU(True),
            nn.ConvTranspose3d(4,1,kernel_size=3,stride=1,padding=1),
        )

    def forward(self,input):

        out1 = self.encoder(input)

        out = self.decoder(out1)

        return out1,out


class AE2D(nn.Module):

    def __init__(self):
        super(AE2D, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, input):
        out1 = self.encoder(input)

        out = self.decoder(out1)

        return out1,out

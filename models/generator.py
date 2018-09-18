from torch import nn
from .basic_model import BasicModel

class NetG(BasicModel):
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.model_name = "NetG"

        self.opt = opt
        ngf = opt.ngf
        self.main = nn.Sequential(
            # (noise_size, 1, 1)
            nn.ConvTranspose2d(opt.noise_size, ngf*8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True),

            # (ngf*8, 4, 4)
            nn.ConvTranspose2d(ngf*8, ngf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),

            # (ngf*4, 8, 8)
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            # (ngf*2, 16, 16)
            nn.ConvTranspose2d(ngf*2, ngf*1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (ngf*1, 32, 32)
            nn.ConvTranspose2d(ngf*1, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh(),

            # (3, 96, 96)
        )

    def forward(self, x):
        return self.main(x)

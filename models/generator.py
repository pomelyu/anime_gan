from torch import nn
from .basic_model import BasicModel

class NetG(BasicModel):
    def __init__(self, opt):
        super(NetG, self).__init__()
        self.model_name = "NetG"

        self.opt = opt
        self.model = nn.Sequential(
            # (noise_size, 1, 1)
            nn.ConvTranspose2d(opt.noise_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # (512, 4, 4)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # (256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (128, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (64, 32, 32)
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),

            # (3, 64, 64)
        )

    def forward(self, input):
        return self.model(input)

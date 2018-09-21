from torch import nn
from .basic_model import BasicModel

class NetD(BasicModel):
    def __init__(self, opt):
        super(NetD, self).__init__()
        self.model_name = "NetD"

        ndf = opt.ndf
        self.main = nn.Sequential(
            # (3, 96, 96)
            nn.Conv2d(3, ndf*1, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(inplace=True),

            # (ndf*1, 32, 32)
            nn.Conv2d(ndf*1, ndf*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(inplace=True),

            # (ndf*2, 16, 16)
            nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(inplace=True),

            # (ndf*4, 8, 8)
            nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(inplace=True),

            # (ndf*8, 4, 4)
            nn.Conv2d(ndf*8, 1, kernel_size=4, stride=1, padding=0, bias=False),

            # (1, 1, 1)
        )

    def forward(self, x):
        return self.main(x).view(-1)

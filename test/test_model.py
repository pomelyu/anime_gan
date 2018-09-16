import torch
from torch.utils.data import DataLoader

import context
from dataset import AnimeData, NoiseData
from models import NetG, NetD

# Test arguments
# ============================= #
DATA_PATH = "data/faces"
BATCH_SIZE = 16
NOISE_SIZE = 100
# ============================= #

# Prepare
noise_data = NoiseData(NOISE_SIZE, BATCH_SIZE)
noise_dataloader = DataLoader(noise_data, batch_size=BATCH_SIZE)
noise_iter = iter(noise_dataloader)

feature_map = next(noise_iter)

class Opt():
    noise_size = NOISE_SIZE

opt = Opt()


# Test
net_G = NetG(opt)
net_D = NetD()

generated = net_G(feature_map)
assert(generated.shape == torch.Tensor(BATCH_SIZE, 3, 64, 64).shape)
assert(torch.max(generated) <= 1)
assert(torch.min(generated) >= 0)

res_generated = net_D(generated)
assert(res_generated.shape == torch.Tensor(BATCH_SIZE).shape)
assert(torch.max(res_generated) <= 1)
assert(torch.max(res_generated) >= 0)

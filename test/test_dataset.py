from torch.utils.data import DataLoader
import torch

import context
from dataset import AnimeData, NoiseData

# Test arguments
# ============================= #
DATA_PATH = "data/faces"
BATCH_SIZE = 16
NOISE_SIZE = 100
# ============================= #

# Test Anime data
anime_data = AnimeData(DATA_PATH)
anime_dataloader = DataLoader(anime_data, batch_size=BATCH_SIZE, shuffle=True)

anime_iter = iter(anime_dataloader)
images = next(anime_iter)

assert(images.shape == torch.Tensor(BATCH_SIZE, 3, 64, 64).shape)
assert(torch.max(images) <= 1)
assert(torch.min(images) >= 0)

# Test Noise data
noise_data = NoiseData(NOISE_SIZE, len(anime_data))
noise_dataloader = DataLoader(noise_data, batch_size=BATCH_SIZE)

noise_iter = iter(noise_dataloader)
noises = next(noise_iter)

assert(noises.shape == torch.Tensor(BATCH_SIZE, NOISE_SIZE, 1, 1).shape)
assert(torch.max(noises) <= 1)
assert(torch.min(noises) >= 0)

# pylint: disable=invalid-name
from torch.utils.data import DataLoader
import torch

import context # pylint: disable=unused-import
from dataset import AnimeData, NoiseData

# ============================= #
# Test arguments
# ============================= #
DATA_PATH = "data/faces"
BATCH_SIZE = 16
NOISE_SIZE = 100


# ============================= #
# Testing
# ============================= #
def test_anime_data():
    anime_data = AnimeData(DATA_PATH)
    anime_dataloader = DataLoader(anime_data, batch_size=BATCH_SIZE, shuffle=True)

    anime_iter = iter(anime_dataloader)
    images = next(anime_iter)

    assert images.shape == torch.Tensor(BATCH_SIZE, 3, 96, 96).shape
    assert torch.max(images) <= 1 and torch.min(images) >= 0


def test_noise_data():
    noise_data = NoiseData(NOISE_SIZE, BATCH_SIZE)
    noise_dataloader = DataLoader(noise_data, batch_size=BATCH_SIZE)

    noise_iter = iter(noise_dataloader)
    noises = next(noise_iter)

    assert noises.shape == torch.Tensor(BATCH_SIZE, NOISE_SIZE, 1, 1).shape
    assert torch.max(noises) <= 1 and torch.min(noises) >= 0


if __name__ == "__main__":
    test_anime_data()
    test_noise_data()

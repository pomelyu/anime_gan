import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

class AnimeData(Dataset):
    def __init__(self, data_path):
        self.images = [os.path.join(data_path, name) for name in os.listdir(data_path) if name.endswith(".jpg")]
        self.transforms = T.Compose([
            T.Resize(96),
            T.CenterCrop(96),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        image_path = self.images[index]
        image = Image.open(image_path)
        data = self.transforms(image)

        return data

    def __len__(self):
        return len(self.images)

class NoiseData(Dataset):
    def __init__(self, noise_size, length):
        self.noise_size = noise_size
        self.length = length

    def __getitem__(self, index):
        return torch.randn(self.noise_size, 1, 1)

    def __len__(self):
        return self.length

import time

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from dataset import NoiseData
from config import opt

class Evaluator():
    def __init__(self, opt, grid_size=5):
        self.out_path = opt.out_path
        self.grid_size = grid_size

        noise_data = NoiseData(opt.noise_size, grid_size*grid_size)
        noise_dataloader = DataLoader(noise_data, batch_size=grid_size*grid_size)
        nosie_iter = iter(noise_dataloader)
        feature_maps = next(nosie_iter)
        if opt.use_gpu:
            feature_maps = feature_maps.cuda()
        self.feature_maps = feature_maps

    def evaluate(self, generator):
        res = generator(self.feature_maps)
        res = res.cpu()
        res = res * torch.FloatTensor([0.5, 0.5, 0.5]).view(3, 1, 1) + torch.FloatTensor([0.5, 0.5, 0.5]).view(3, 1, 1)
        image = make_grid(res, self.grid_size)

        prefix = "{}/result_".format(self.out_path)
        image_path = time.strftime(prefix + "%m%d_%H_%M_%S.jpg")
        save_image(image, image_path)


if __name__ == "__main__":
    from models import NetG

    net_G = NetG(opt)
    evaluator = Evaluator(opt, grid_size=5)
    evaluator.evaluate(net_G)

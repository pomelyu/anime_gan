import time

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from dataset import NoiseData

class Evaluator():
    def __init__(self, out_dir, noise_size, grid_size=5):
        self.out_dir = out_dir
        self.grid_size = grid_size

        noise_data = NoiseData(noise_size, grid_size*grid_size)
        noise_dataloader = DataLoader(noise_data, batch_size=grid_size*grid_size)
        nosie_iter = iter(noise_dataloader)
        self.feature_maps = next(nosie_iter)

    def evaluate(self, generator):
        res = generator(self.feature_maps)
        image = make_grid(res, self.grid_size)

        prefix = "{}/result_".format(self.out_dir)
        image_path = time.strftime(prefix + "%m%d_%H_%M_%S.jpg")
        save_image(image, image_path)


if __name__ == "__main__":
    from models import NetG

    class Opt():
        noise_size = 100

    opt = Opt()

    net_G = NetG(opt)
    evaluator = Evaluator("out", opt.noise_size, grid_size=5)
    evaluator.evaluate(net_G)

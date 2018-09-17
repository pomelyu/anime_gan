# pylint: disable=invalid-name
import torch
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

from dataset import AnimeData, NoiseData
from models import NetG, NetD
from evaluator import Evaluator
from config import opt

def train(**kwargs):
    opt._parse(kwargs)

    demoer = Evaluator(opt.out_path, opt.noise_size)

    anime_data = AnimeData(opt.data_path)
    anime_dataloader = DataLoader(anime_data, batch_size=opt.batch_size, shuffle=True)

    noise_data = NoiseData(opt.noise_size, len(anime_data))
    noise_dataloader = DataLoader(noise_data, batch_size=opt.batch_size)

    net_G = NetG(opt)
    net_D = NetD()

    criterion = torch.nn.BCELoss()
    optimizer_G = torch.optim.Adam(net_G.parameters(), lr=opt.lr)
    optimizer_D = torch.optim.Adam(net_D.parameters(), lr=opt.lr)

    loss_D_meteor = meter.AverageValueMeter()
    loss_G_meteor = meter.AverageValueMeter()

    for epoch in range(opt.max_epochs):
        loss_D_meteor.reset()
        loss_G_meteor.reset()

        num_batch = len(anime_dataloader)
        for true_image, feature_map in tqdm(zip(anime_dataloader, noise_dataloader), total=num_batch):
            num_data = true_image.shape[0]

            fake_image = net_G(feature_map)
            fake_score = net_D(fake_image)
            true_score = net_D(true_image)

            # Train discriminator
            optimizer_D.zero_grad()
            net_G.eval()
            net_G.set_requires_grad(False)
            net_D.train()
            net_D.set_requires_grad(True)

            loss_D = criterion(fake_score, torch.zeros(num_data)) + \
                criterion(true_score, torch.ones(num_data))
            loss_D.backward(retain_graph=True)
            optimizer_D.step()

            loss_D_meteor.add(loss_D.detach().item())

            # Train generator
            optimizer_G.zero_grad()
            net_G.train()
            net_G.set_requires_grad(True)
            net_D.eval()
            net_D.set_requires_grad(False)

            loss_G = criterion(fake_score, torch.ones(num_data))
            loss_G.backward()
            optimizer_G.step()

            loss_G_meteor.add(loss_G.detach().item())

        print("Epoch {epoch:0>2d}: loss_D - {loss_D:.3f}, loss_G - {loss_G:.3f}".format(
            epoch=epoch+1,
            loss_D=loss_D_meteor.value()[0],
            loss_G=loss_G_meteor.value()[0],
        ))

        demoer.evaluate(net_G)
        net_D.save("checkpoints")
        net_G.save("checkpoints")

if __name__ == "__main__":
    import fire
    fire.Fire()

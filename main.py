# pylint: disable=invalid-name
import time
import os
import torch
from torch.utils.data import DataLoader
from torchnet import meter
from tqdm import tqdm

from dataset import AnimeData, NoiseData
from models import NetG, NetD
from models.loss import WGANLoss
from evaluator import Evaluator
from config import opt

def train(**kwargs):
    opt._parse(kwargs)

    demoer = Evaluator(opt)

    anime_data = AnimeData(opt.data_path)
    anime_dataloader = DataLoader(anime_data, batch_size=opt.batch_size, shuffle=True)

    noise_data = NoiseData(opt.noise_size, len(anime_data))
    noise_dataloader = DataLoader(noise_data, batch_size=opt.batch_size, shuffle=True)

    net_G = NetG(opt)
    net_D = NetD(opt)

    criterion = WGANLoss(opt.wgan_lambda, opt.use_gpu)

    if opt.use_gpu:
        net_G = net_G.cuda()
        net_D = net_D.cuda()
        criterion = criterion.cuda()

    optimizer_G = torch.optim.RMSprop(net_G.parameters(), lr=opt.lr_g, momentum=opt.beta1, alpha=opt.beta2)
    optimizer_D = torch.optim.RMSprop(net_D.parameters(), lr=opt.lr_d, momentum=opt.beta1, alpha=opt.beta2)

    loss_D_meteor = meter.AverageValueMeter()
    loss_G_meteor = meter.AverageValueMeter()

    if opt.netd_path is not None:
        net_D.load(opt.netd_path)
    if opt.netg_path is not None:
        net_G.load(opt.netg_path)

    for epoch in range(opt.max_epochs):
        loss_D_meteor.reset()
        loss_G_meteor.reset()

        num_batch = len(anime_dataloader)
        generator = enumerate(zip(anime_dataloader, noise_dataloader))
        for ii, (true_image, feature_map) in tqdm(generator, total=num_batch, ascii=True):

            if opt.use_gpu:
                feature_map = feature_map.cuda()
                true_image = true_image.cuda()

            net_G.set_requires_grad(False)
            net_D.set_requires_grad(True)

            # Train discriminator
            if ii % opt.every_d == 0:
                optimizer_D.zero_grad()

                fake_image = net_G(feature_map)
                fake_score = net_D(fake_image)
                true_score = net_D(true_image)
                loss_D = criterion.discriminator_loss(net_D, true_score, fake_score, true_image, fake_image)
                loss_D.backward()
                optimizer_D.step()

                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()


            net_G.set_requires_grad(True)
            net_D.set_requires_grad(False)

            loss_D_meteor.add(loss_D.detach().item())

            # Train generator
            if ii % opt.every_g == 0:
                optimizer_G.zero_grad()

                fake_image = net_G(feature_map)
                fake_score = net_D(fake_image)
                loss_G = criterion.generator_loss(fake_score)
                loss_G.backward()
                optimizer_G.step()

                loss_G_meteor.add(loss_G.detach().item())

                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        gan_log = "Epoch {epoch:0>2d}: loss_D - {loss_D}, loss_G - {loss_G}".format(
            epoch=epoch+1,
            loss_D=loss_D_meteor.value()[0],
            loss_G=loss_G_meteor.value()[0],
        )
        print(gan_log)

        if epoch % opt.save_freq == opt.save_freq - 1:
            demoer.evaluate(net_G)
            net_D.save(opt.save_model_path)
            net_G.save(opt.save_model_path)
        time.sleep(0.5)

if __name__ == "__main__":
    import fire
    fire.Fire()

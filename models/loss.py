import torch.nn as nn
import torch

# Define the GAN loss base on Wasserstein Metrix
# Base on "Improved Training of Wasserstein GANs" i.e. WGAN-GP
# https://arxiv.org/abs/1704.00028
class WGANLoss(nn.Module):
    def __init__(self, lambda_gp, use_gpu):
        super(WGANLoss, self).__init__()
        self.lambda_gp = lambda_gp
        self.use_gpu = use_gpu

    def discriminator_loss(self, discriminator, real_label, fake_label, real_data, fake_data):
        assert real_data.shape == fake_data.shape
        n = real_data.shape[0]
        # randomly choose the iterpolation between real_data and fake_data
        inter = torch.rand(n, 1, 1, 1)
        if self.use_gpu:
            inter = inter.cuda()
        inter_data = inter * real_data + (1 - inter) * fake_data

        # calculate dD(x)/dx
        inter_data.requires_grad = True
        inter_label = discriminator(inter_data)
        grad_outpus = torch.ones(inter_label.shape, dtype=torch.float, requires_grad=False)
        if self.use_gpu:
            grad_outpus = grad_outpus.cuda()
        gradients = torch.autograd.grad(inter_label, inter_data, grad_outputs=grad_outpus, only_inputs=True)[0]
        gradients = gradients.view(n, -1)

        # penalty = E[(||dD(x)/dx|| - 1)^2]
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return fake_label.mean() - real_label.mean() + self.lambda_gp * gradient_penalty

    def generator_loss(self, fake_label):
        return -fake_label.mean()

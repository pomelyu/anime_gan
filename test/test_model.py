import torch
from torch.utils.data import DataLoader
import numpy as np

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

net_G = NetG(opt)
net_D = NetD()

criterion = torch.nn.MSELoss()
optimzer = torch.optim.SGD(net_D.parameters(), lr=0.1)

# Test
generated = net_G(feature_map)
assert(generated.shape == torch.Tensor(BATCH_SIZE, 3, 64, 64).shape)
assert(torch.max(generated) <= 1)
assert(torch.min(generated) >= 0)

res_generated = net_D(generated)
assert(res_generated.shape == torch.Tensor(BATCH_SIZE).shape)
assert(torch.max(res_generated) <= 1)
assert(torch.max(res_generated) >= 0)

# Test: freeze params
net_D.set_requires_grad(False)
ori_params = []
for param in net_D.parameters():
    ori_params.append(param.detach().numpy().copy())
    assert param.requires_grad == False

loss = criterion(res_generated, torch.ones(BATCH_SIZE, dtype=torch.float32))
loss.backward()
optimzer.step()

for i, param in enumerate(net_D.parameters()):
    assert np.array_equal(param.detach().numpy(), ori_params[i])


# Test: Unfreeze params
ori_params = []
net_D.set_requires_grad(True)
for param in net_D.parameters():
    ori_params.append(param.detach().numpy().copy())
    assert param.requires_grad == True

generated = net_G(feature_map)
res_generated = net_D(generated)
loss = criterion(res_generated, torch.ones(BATCH_SIZE, dtype=torch.float32))
loss.backward()
optimzer.step()

for i, param in enumerate(net_D.parameters()):
    assert np.array_equal(param.detach().numpy(), ori_params[i]) == False

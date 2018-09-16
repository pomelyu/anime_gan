import torch
import time

class BasicModel(torch.nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()
        self.model_name = str(type(self))
        self.model = torch.nn.Sequential()

    def save(self, checkpoint_dir):
        if self.model_namename is None:
            prefix = "{}/{}_".format(checkpoint_dir, self.model_name)
            name = time.strftime(prefix + "%m%d_%H_%M_%S.pth")

        torch.save(self.state_dict(), name)
        return name

    def load(self, model_path):
        self.load_state_dict(torch.load(model_path))

    def set_requires_grad(self, requires_grad=False):
        if self.model is not None:
            for param in self.model.parameters():
                param.requires_grad = requires_grad

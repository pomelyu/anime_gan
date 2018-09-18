import warnings

class DefaultConfig(object):
    use_gpu = False
    noise_size = 100
    ngf = 64
    ndf = 64

    data_path = "data/faces"
    save_model_path = "checkpoints"
    out_path = "out"

    netd_path = None
    netg_path = None

    max_epochs = 200
    batch_size = 256
    lr_d = 2e-4
    lr_g = 2e-4
    beta1 = 0.5
    beta2 = 0.999

    every_d = 1
    every_g = 5

    save_freq = 10

    def _parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has no attribute", k)
            else:
                setattr(self, k, v)

        print("use config")
        for k, _ in self.__class__.__dict__.items():
            if not k.startswith("_"):
                print(k, ":", getattr(self, k))

opt = DefaultConfig()

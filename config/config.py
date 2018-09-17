import warnings

class DefaultConfig(object):
    use_gpu = False
    noise_size = 100

    data_path = "data/faces"
    save_model_path = "checkpoints"
    out_path = "out"

    max_epochs = 5
    batch_size = 16
    lr = 0.001

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

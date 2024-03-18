import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.group_name = "mppi"

    config.model_cls = "MPPILearner"

    config.dynamics_lr = 3e-4
    config.hidden_dims = (256, 256)
    config.dynamics_ensemble_size = None
    config.beta = 2.0
    config.discount = 0.99
    config.ema = 0.5
    config.ema_discount = 0.9
    config.horizon = 10
    config.num_rollouts = 128
    config.sampling_std = 0.3

    return config

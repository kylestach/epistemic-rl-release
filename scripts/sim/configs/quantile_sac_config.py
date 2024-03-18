import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.model_cls = "QuantileSACLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.critic_hidden_dims = (512, 512, 512)
    config.actor_hidden_dims = (256, 256)

    config.discount = 0.95

    config.num_qs = 10
    # config.num_min_qs = 2

    config.tau = 0.005
    config.init_temperature = 1.0
    config.target_entropy = config_dict.placeholder(float)

    config.num_quantiles = 11
    config.cvar_risk = 0.9

    config.critic_weight_decay = 1e-3
    config.critic_layer_norm = True

    config.backup_entropy = False
    config.independent_ensemble = True

    return config

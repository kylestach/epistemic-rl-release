import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.group_name = "gaussian_cvar"

    config.model_cls = "GaussianDistributionalSACLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.actor_hidden_dims = (256, 256)
    config.critic_hidden_dims = (256, 256)

    config.discount = 0.99

    config.num_qs = 1

    config.tau = 0.005
    config.init_temperature = 1e-3
    config.target_entropy = config_dict.placeholder(float)
    config.critic_weight_decay = 1e-3
    config.critic_layer_norm = True

    config.backup_entropy = False

    # beta is hardcoded

    return config

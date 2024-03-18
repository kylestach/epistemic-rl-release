import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.group_name = "sac_safety"

    config.model_cls = "SafetySACLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4
    config.safety_lagrange_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.num_qs = 10
    config.num_min_qs = 2

    config.tau = 0.005
    config.init_temperature = 1.0
    config.target_entropy = config_dict.placeholder(float)
    config.critic_weight_decay = 1e-2
    config.critic_layer_norm = True
    config.init_safety_lagrange = 10.0

    config.backup_entropy = False

    config.target_safety = 0.001

    return config

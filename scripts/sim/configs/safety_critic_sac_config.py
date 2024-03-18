import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.group_name = "safety_critic"

    config.model_cls = "SafetyCriticSACLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4

    config.hidden_dims = (256, 256)

    config.discount = 0.99

    config.num_qs = 2

    config.tau = 0.005
    config.init_temperature = 1e-3
    config.target_entropy = config_dict.placeholder(float)
    config.critic_weight_decay = 1e-2
    config.critic_layer_norm = True

    config.backup_entropy = False

    config.safety_threshold = 0.1
    config.safety_discount = 0.7

    return config

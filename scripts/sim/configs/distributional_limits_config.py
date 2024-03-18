import ml_collections
from ml_collections.config_dict import config_dict


def get_config():
    config = ml_collections.ConfigDict()

    config.group_name = "distributional_limits"

    config.model_cls = "DistributionalSACLearner"

    config.actor_lr = 3e-4
    config.critic_lr = 3e-4
    config.temp_lr = 3e-4
    config.limits_lr = 1e-5
    config.q_entropy_lagrange_lr = 1e-3

    config.critic_hidden_dims = (256, 256)
    config.actor_hidden_dims = (256, 256)

    config.discount = 0.99

    config.num_qs = 2

    config.tau = 0.005
    config.init_temperature = 1.0
    config.target_entropy = config_dict.placeholder(float)

    config.num_atoms = 151
    config.q_min = -100.0
    config.q_max = 650.0
    config.cvar_risk = 0.9

    config.critic_weight_decay = 1e-3
    config.critic_layer_norm = True

    config.limits_weight_decay = 1e-3

    config.backup_entropy = False
    config.independent_ensemble = True

    config.cvar_limits = "cvar_risk"
    config.q_entropy_target_diff = -0.01

    config.q_entropy_lagrange_init = 1e-3

    config.learned_action_space_idx = 1
    config.learned_action_space_initial_value = -0.5

    return config

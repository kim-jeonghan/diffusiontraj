def _get_config_value(config, key, default=None, required=False):
    if key in config:
        return config[key]
    if required:
        raise KeyError(f"Missing config key: {key}")
    return default


def normalize_trajectory_stitching_policy_config(policy_config):
    inference_schedule = _get_config_value(
        policy_config,
        "inference_schedule",
        default="interleaved",
    )
    inference_schedule_aliases = {
        "gsc": "global_sync",
    }
    inference_schedule = inference_schedule_aliases.get(
        inference_schedule, inference_schedule
    )
    return {
        "num_segments": _get_config_value(policy_config, "num_segments", required=True),
        "top_k": _get_config_value(policy_config, "top_k", required=True),
        "trajectory_selection": _get_config_value(
            policy_config, "trajectory_selection", required=True
        ),
        "inference_schedule": inference_schedule,
    }


def normalize_trajectory_blender_config(blender_config):
    return {
        "blend_type": _get_config_value(blender_config, "blend_type", required=True),
        "blend_exponential_beta": _get_config_value(
            blender_config,
            "blend_exponential_beta",
            default=3,
        ),
    }


def normalize_maze_policy_config(policy_config, default_plan_horizon):
    return {
        "plan_horizon": _get_config_value(
            policy_config,
            "plan_horizon",
            default=default_plan_horizon,
        ),
    }


def extract_planner_runtime_config(args):
    args_dict = vars(args)
    policy_config = {}

    for key in (
        "num_segments",
        "top_k",
        "trajectory_selection",
        "inference_schedule",
        "plan_horizon",
    ):
        if key in args_dict:
            policy_config[key] = args_dict[key]

    blender_config = {}
    if "blend_type" in args_dict:
        blender_config["blend_type"] = args_dict["blend_type"]

    if "blend_exponential_beta" in args_dict:
        blender_config["blend_exponential_beta"] = args_dict["blend_exponential_beta"]

    return policy_config, blender_config

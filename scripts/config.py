BOTTLE_FLIP_TASK_ARGS = {
    "env_name": "BottleFlipTask",
    "robots": ["Panda"],
    "has_offscreen_renderer": False,
    "use_camera_obs": False,
    "use_object_obs": True,
    "render_camera": None,
    "control_freq": 20,
    "reward_shaping": True,
    "horizon": 500,
    "hard_reset": True,
}

MODEL_PATH = "./models/"
FILENAME = "sac_bottle_flip"

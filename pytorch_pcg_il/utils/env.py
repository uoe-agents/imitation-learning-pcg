import gym
from gym.envs.registration import register # do not delete --> used for procgen envs

def make_env_minigrid(env_dict, seed=None):
    env_key = list(env_dict.keys())[0]
    env = gym.make(env_key)
    env.seed(seed)
    return env

def make_env_procgen(env_dict, num_levels, start_level=0,distribution_mode='easy'):
    """
    render = False
    render_mode ='human'

    if render:
        render_mode = "human"

    use_viewer_wrapper = False
    kwargs["render_mode"] = render_mode
    if render_mode == "human":
        # procgen does not directly support rendering a window
        # instead it's handled by gym3's ViewerWrapper
        # procgen only supports a render_mode of "rgb_array"
        use_viewer_wrapper = True
        kwargs["render_mode"] = "rgb_array"
    """

    # kwargs = {
    #     "env_name":'coinrun',
    #     "paint_vel_info": False,
    #     "use_generated_assets": False,
    #     "center_agent": False,
    #     "use_backgrounds": False,
    #     "restrict_themes": False,
    #     "use_monochrome_assets": False,
    # }
    # use_viewer_wrapper = True

    env_name = env_key = list(env_dict.keys())[0] #also known as
    env = gym.make("procgen:procgen-{}-v0".format(env_name),start_level=start_level,num_levels=num_levels,distribution_mode=distribution_mode)
    gym_env=env
    return gym_env


ENV_NAMES = [
    # "bigfish",
    # "bossfight",
    # "caveflyer",
    # "chaser",
    "climber",
    # "coinrun",
    # "dodgeball",
    # "fruitbot",
    # "heist",
    # "jumper",
    # "leaper",
    # "maze",
    # "miner",
    "ninja",
    # "plunder",
    # "starpilot",
]

def register_environments_procgen():
    for env_name in ENV_NAMES:
        register(
            id=f'procgen-{env_name}-v0',
            entry_point='procgen.gym_registration:make_env',
            kwargs={"env_name": env_name},
        )
        print('registered:',env_name)
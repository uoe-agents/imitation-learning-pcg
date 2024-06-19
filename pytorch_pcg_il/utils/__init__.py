from .env import *
from .format import *
from .other import *
from .storage import *

# trigger registrations
import gym_minigrid # do not delete --> triggers registration of envs
register_environments_procgen() # triggers procgen envs registration
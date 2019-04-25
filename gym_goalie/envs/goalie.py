import os
import numpy as np

from gym import utils
from gym_goalie.envs import goalie_env


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join('fetch', 'goalie.xml')


class GoalieTestEnv(goalie_env.GoalieEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.05,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.7, 1.1, 0.4, 1., 0., 0., 0.],
        }
        goalie_env.GoalieEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,  # 20 steps until update of rewards
            gripper_extra_height=-0.02, target_in_the_air=False, target_offset=0.0,
            obj_range=0.1, target_range=0.3, distance_threshold=0.5,  # !!! may be super relevant!
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

from gym.envs.registration import register

register(
    id='Goalie-v0',
    entry_point='gym_goalie.envs:GoalieTestEnv',
    kwargs={'reward_type': 'dense'},
    max_episode_steps=100
)

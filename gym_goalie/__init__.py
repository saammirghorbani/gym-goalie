from gym.envs.registration import register

register(
    id='Goalie-v0',
    entry_point='gym_goalie.envs:GoalieTestEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=100
)

register(
    id='Goalie-long-v0',
    entry_point='gym_goalie.envs:GoalieTestEnv',
    kwargs={'reward_type': 'sparse'},
    max_episode_steps=2000
)

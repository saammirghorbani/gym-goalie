from gym.envs.registration import register

register(
    id='Goalie-v0',
    entry_point='gym_goalie.envs:GoalieEnv',
)

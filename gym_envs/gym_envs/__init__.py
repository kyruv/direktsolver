from gym.envs.registration import register

register(
    id='UnityEnv-v0',
    entry_point='gym_envs.envs:UnityEnv_v0',
    max_episode_steps=300,
)
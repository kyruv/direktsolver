from gymnasium.envs.registration import register

register(
    id='direkt-v0',
    entry_point='gym_envs.direkt.direkt_v0:Direkt_v0',
)
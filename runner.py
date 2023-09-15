import gym_envs
import gymnasium as gym

class Runner:

    def __init__(self):
        self.env = self.env = gym.make("direkt-v0", render_mode="human")
        self.env.reset()

    def spin(self):

        while True:
            print("hello")


Runner().spin()
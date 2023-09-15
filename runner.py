import gym_envs
import gymnasium as gym
import pygame

class Runner:

    def __init__(self):
        self.env = self.env = gym.make("direkt-v0", render_mode="human", level="levels/level1.json")
        self.env.reset()

    def spin(self):

        while True:
            while True:
                events = pygame.event.get()
                self.env.render()

                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            return
                        
                        if event.key == pygame.K_RIGHT:
                            obs, _, _, _, _ = self.env.step(0)
                        if event.key == pygame.K_UP:
                            obs, _, _, _, _ = self.env.step(1)
                        if event.key == pygame.K_LEFT:
                            obs, _, _, _, _ = self.env.step(2)
                        if event.key == pygame.K_DOWN:
                            obs, _, _, _, _ = self.env.step(3)
                        if event.key == pygame.K_SPACE:
                            obs, _, _, _, _ = self.env.step(4)
                        


Runner().spin()
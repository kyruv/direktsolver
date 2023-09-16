import gym_envs
import gymnasium as gym
import pygame

class Runner:

    def __init__(self):
        self.env = self.env = gym.make("direkt-v0", render_mode="human", level="levels/level4.json")
        self.env.reset()

    def spin(self):

        for i in range(100):
            reward = 0
            terminated = False
            self.env.reset()
            while not terminated:
                events = pygame.event.get()
                self.env.render()

                for event in events:
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            return
                        
                        if event.key == pygame.K_RIGHT:
                            _, r, terminated, _, _ = self.env.step(0)
                        if event.key == pygame.K_DOWN:
                            _, r, terminated, _, _ = self.env.step(1)
                        if event.key == pygame.K_LEFT:
                            _, r, terminated, _, _ = self.env.step(2)
                        if event.key == pygame.K_UP:
                            _, r, terminated, _, _ = self.env.step(3)
                        if event.key == pygame.K_x:
                            _, r, terminated, _, _ = self.env.step(4)
                        if event.key == pygame.K_z:
                            _, r, terminated, _, _ = self.env.step(6)
                        if event.key == pygame.K_w:
                            _, r, terminated, _, _ = self.env.step(5)
                        
                        reward += r
                
            print(f"Episode {i} over with {reward} reward.")
                        


Runner().spin()
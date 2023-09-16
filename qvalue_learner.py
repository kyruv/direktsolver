import numpy as np
import os
import json
import gym_envs
import gymnasium as gym
from datetime import datetime
import math

class Runner:

    def __init__(self, level, load=True, overwrite=False):
        self.env = gym.make("direkt-v0", render_mode=None, level=f"levels/{level}.json")

        dir = os.path.dirname(os.path.realpath(__file__))
        self.model_path = os.path.join(dir, f"models/{level}.npy")
        level_path = os.path.join(dir, f"gym_envs/direkt/levels/{level}.json")
        ld = json.load(open(level_path))
        self.num_enemies = len(ld["slow_enemies"]) + len(ld["fast_enemies"])
        self.num_gates = len(ld["gates"])

        if load:
            self.q_table = np.load(self.model_path)
        else:
            already_exists = os.path.exists(self.model_path)
            if not overwrite and already_exists:
                raise Exception("You are about to erase existing trained data")
            
            
            dim = np.array(ld["level_setup"]).shape

            # q table needs entire state which is
            #   player r, player c
            #   enemy1 r, enemy1 c, enemy1 rot ... enemyN r, enemyN c, enemyN rot 
            #   gate1 rotation [0,3], ... gateN rotation [0,3]
            #   player action [0,6]
            shape = []
            shape.append(dim[0])
            shape.append(dim[1])
            for _ in range(self.num_enemies):
                shape.append(dim[0])
                shape.append(dim[1])
                shape.append(4)
            for _ in range(self.num_gates):
                shape.append(4)
            shape.append(7)

            self.q_table = np.zeros(shape)
            np.save(self.model_path, self.q_table)
            
        

    
    def train(self):
        # training hyper params
        alpha = .01
        bias_best = 0
        num_episodes = 10000
        gamma = .95
        max_epsilon = .25
        max_episode_steps = 100

        best_found_reward = -math.inf
        best_history = ([], 0)

        if os.path.exists(f'{self.model_path}_solution.csv'):
            with open(f'{self.model_path}_solution.csv', 'rb') as f:
                try:  # catch OSError in case of a one line file 
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                last_line = f.readline().decode()
                data = last_line.split(',')
                best_found_reward = int(data[3])
        
        for episode in range(num_episodes):
            
            epsilon = np.power(1 - episode / num_episodes,2) * max_epsilon
            state, _ = self.env.reset()

            terminated = False
            reward = 0
            actions = 0
            episode_action_list = []
            history = []
            
            while not terminated and actions < max_episode_steps:
                # 1. decide action
                action = self._get_greedy_action(state, epsilon)

                # 2. take action
                new_state, r, terminated, _, _ = self.env.step(action)
                reward += r
                actions += 1
                episode_action_list.append(action)
                if actions == max_episode_steps: # an additional penalty for running out of time
                    reward -= 100

                # print(f"p {state['player']} a {action} ns {new_state['player']}" )

                # 3. save history for update later. 
                history.append((state, action, new_state))

                # 4. update state to wherever we went
                state = new_state
            
            # don't care if you fail quick or fail slow
            if reward < 0:
                reward = -100

            if reward > best_found_reward:
                best_history = (history, reward)
                best_found_reward = reward
                with open(f'{self.model_path}_solution.csv','a') as fd:
                    episode_action_str = '.'.join([str(a) for a in episode_action_list])
                    fd.write(','.join([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(episode), str(epsilon), str(reward), episode_action_str]) + '\n')
            
            # standard q learning update - learn from our last episode
            for (s, a, ns) in history:

                _, best_score = self._get_best_action_from(ns)
                old_q_value = self._q_table(s, a)
                updated_q_value = old_q_value + alpha * (reward + gamma * best_score - old_q_value)
                self._q_table_update(s, a, updated_q_value)

            # bias towards our best solution by retraining on it
            # print(f"doing biasing on best solution towards {best_history[1]}")
            for (s, a, ns) in best_history[0]:
                _, best_score = self._get_best_action_from(ns)
                old_q_value = self._q_table(s, a)
                updated_q_value = old_q_value + bias_best * alpha * (best_history[1] + gamma * best_score - old_q_value)
                self._q_table_update(s, a, updated_q_value)

            if episode % 10 == 0 or episode == num_episodes:
                np.save(self.model_path, self.q_table)

                with open(f'{self.model_path}_progress.csv','a') as fd:
                    fd.write(','.join([datetime.now().strftime('%Y-%m-%d %H:%M:%S'), str(episode), str(epsilon), str(reward), str(actions)]) + '\n')
    
    # helper function to access q_table
    def _q_table(self, state, action):
        t = self.q_table
        # plug in player
        t = t[state["player"][0]]
        t = t[state["player"][1]]

        # plug in enemies
        for i in range(self.num_enemies):
            t = t[state["enemies"]["location"][2*i]] 
            t = t[state["enemies"]["location"][2*i+1]]
            t = t[state["enemies"]["rotation"][i]]
        
        # plug in gates
        for i in range(self.num_gates):
            t = t[state["gates"]["rotation"][i]]
        
        # plug in action
        return t[action]

    # helper function to access q_table
    def _q_table_update(self, state, action, val):
        t = self.q_table
        
        # plug in player
        t = t[state["player"][0]]
        t = t[state["player"][1]]

        # plug in enemies
        for i in range(self.num_enemies):
            t = t[state["enemies"]["location"][2*i]] 
            t = t[state["enemies"]["location"][2*i+1]]
            t = t[state["enemies"]["rotation"][i]]
        
        # plug in gates
        for i in range(self.num_gates):
            t = t[state["gates"]["rotation"][i]]
        
        # plug in action and update
        t[action] = val
        
    def _get_greedy_action(self, state, epsilon):
        if np.random.uniform(0,1) < epsilon:
            return np.random.choice(self.env.get_valid_actions())
    
        best_action, _ = self._get_best_action_from(state)

        if state["player"][0] == 1 and state["player"][1] == 0:
            print(f"best action is {best_action}")

        return best_action

    def _get_best_action_from(self, state):
        best_action = None
        best_score = None
        for action in range(7):
            action_score = self._q_table(state, action)
            if best_score == None or action_score > best_score:
                best_score = action_score
                best_action = action
        return best_action, best_score


Runner(level="level4", load=False, overwrite=True).train()
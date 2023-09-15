import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
import pygame
import os

class Direkt_v0(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None, level=None):
        self.level = level
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode
        self.window = None
        self.size = 512
        self.window_size = 512
        self.clock = None
        self.level_file = level

        self.level = Level(level)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.level.reset()
        if self.render_mode == "human":
            self._render_frame()

        return 0, {}

    def _getobs(self, data):
        return data

    def reward(self, obs):
        return -1

    def step(self, action):
        terminated = False
        reward = -10
        # rotating is 0-time action, only mildly punish it
        if action == 4:
            reward = -1

        if action in self.level.get_valid_actions():
            result = self.level.take_action(action)
            terminated = result != 0
            if result == -1:
                reward = -1000
            elif result == 1:
                reward = 1000

        return 0, reward, terminated, False, {}

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()
    
    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        
        # draw walkable tiles
        for row in self.level.location_objects:
            for l in row:
                if l == None:
                    continue

                r = l.draw_loc[0]
                c = l.draw_loc[1]
                color = (200,200,200)
                if l.is_goal:
                    color = (255,0,0)
                pygame.draw.rect(canvas, color, (25*c, 25*r, 23,23))

        # draw gates       
        for row in self.level.location_objects:
            for l in row:
                if l == None:
                    continue

                r = l.draw_loc[0]
                c = l.draw_loc[1]
                if l.gate is not None:
                    color = (0,0,0)
                    tl = (25*c+1, 25*r+1)
                    if l.gate.directions_blocked == [3,0]:
                        p1 = tl
                        p2 = tuple(np.add(tl, (23,0)))
                        p3 = tuple(np.add(tl, (23,23)))
                    elif l.gate.directions_blocked == [2,3]:
                        p1 = tuple(np.add(tl, (23,0)))
                        p2 = tl
                        p3 = tuple(np.add(tl, (0,23)))
                    if l.gate.directions_blocked == [1,2]:
                        p1 = tuple(np.add(tl, (23,23)))
                        p2 = tuple(np.add(tl, (0,23)))
                        p3 = tl
                    elif l.gate.directions_blocked == [0,1]:
                        p1 = tuple(np.add(tl, (23,0)))
                        p2 = tuple(np.add(tl, (23,23)))
                        p3 = tuple(np.add(tl, (0,23)))

                    pygame.draw.lines(canvas, color, False, (p1,p2,p3), 3)
        
        # print("render " + str(self.level.player.location.draw_loc))
        pygame.draw.circle(canvas, (255,255,255), (25*self.level.player.location.draw_loc[1]+12.5, 25*self.level.player.location.draw_loc[0]+12.5), 10)

        for enemy in self.level.slow_enemies:
            center = (25*enemy.location.draw_loc[1]+12.5, 25*enemy.location.draw_loc[0]+12.5)
            if enemy.direction == 2:
                p1 = tuple(np.add(center, (-10,0)))
                p2 = tuple(np.add(center, (8,-8)))
                p3 = tuple(np.add(center, (8,8)))
            elif enemy.direction == 3:
                p1 = tuple(np.add(center, (0,-10)))
                p2 = tuple(np.add(center, (-8,8)))
                p3 = tuple(np.add(center, (8,8)))
            if enemy.direction == 0:
                p1 = tuple(np.add(center, (10,0)))
                p2 = tuple(np.add(center, (-8,8)))
                p3 = tuple(np.add(center, (-8,-8)))
            elif enemy.direction == 1:
                p1 = tuple(np.add(center, (0,10)))
                p2 = tuple(np.add(center, (8,-8)))
                p3 = tuple(np.add(center, (-8,-8)))

            pygame.draw.polygon(canvas, (60,60,60), (p1,p2,p3))

        for enemy in self.level.fast_enemies:
            center = (25*enemy.location.draw_loc[1]+12.5, 25*enemy.location.draw_loc[0]+12.5)
            if enemy.direction == 0:
                p1 = tuple(np.add(center, (-10,0)))
                p2 = tuple(np.add(center, (8,-8)))
                p3 = tuple(np.add(center, (8,8)))
            elif enemy.direction == 1:
                p1 = tuple(np.add(center, (0,-10)))
                p2 = tuple(np.add(center, (-8,8)))
                p3 = tuple(np.add(center, (8,8)))
            if enemy.direction == 2:
                p1 = tuple(np.add(center, (10,0)))
                p2 = tuple(np.add(center, (-8,8)))
                p3 = tuple(np.add(center, (-8,-8)))
            elif enemy.direction == 3:
                p1 = tuple(np.add(center, (0,10)))
                p2 = tuple(np.add(center, (8,-8)))
                p3 = tuple(np.add(center, (-8,-8)))

            pygame.draw.polygon(canvas, (120,120,120), (p1,p2,p3))


        temp_surf = canvas.copy()
        canvas.fill((255,255,255))
        canvas.blit(temp_surf, (50, 50))

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(60)
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )


class Level:

    def __init__(self, level_sheet):
        self.level_sheet = level_sheet
        self.reset()
    
    def reset(self):
        self.fast_enemies = []
        self.slow_enemies = []
        self.player = None
        self.location_objects = None
        self.init_level(self.level_sheet)

    def init_level(self, level_sheet):
        f = open(os.path.join(os.path.dirname(os.path.realpath(__file__)), level_sheet))
        data = json.load(f)

        locations = data["level_setup"]
        location_objects = [[None for i in range(len(locations[0]))] for j in range(len(locations))]
        for r in range(len(locations)):
            for c in range(len(locations[0])):
                if locations[r][c] == 1 or locations[r][c] == 9:
                    location_objects[r][c] = Location(is_goal=locations[r][c] == 9, draw_loc=(r,c))
        
        for r in range(len(locations)):
            for c in range(len(locations[0])):
                if locations[r][c] == 0:
                    continue

                # can have a east
                if c + 1 < len(locations[0]):
                    location_objects[r][c].neighbors[0] = location_objects[r][c + 1]
                
                # can have a south
                if r + 1 < len(locations):
                    location_objects[r][c].neighbors[1] = location_objects[r+1][c]
                
                # can have a west
                if c - 1 >= 0:
                    location_objects[r][c].neighbors[2] = location_objects[r][c - 1]

                # can have a north
                if r - 1 >= 0:
                    location_objects[r][c].neighbors[3] = location_objects[r-1][c]
        
        gates = data["gates"]
        for r, c, init_orientation in gates:
            gate = Gate(init_orientation)
            location_objects[r][c].gate = gate
        
        triggers = data["triggers"]
        for r, c, gr, gc, exit_dir_that_rotates_once in triggers:
            gate = location_objects[gr][gc]
            location = location_objects[r][c]
            m = {
                exit_dir_that_rotates_once: 1,
                exit_dir_that_rotates_once + 2 % 4: 3
            }
            location.exit_trigger_map = m
        
        slow_enemies = data["slow_enemies"]
        for r, c, direction in slow_enemies:
            enemy = Enemy(is_fast=False, direction=direction, location=location_objects[r][c])
            self.slow_enemies.append(enemy)
        
        fast_enemies = data["fast_enemies"]
        for r, c, direction in fast_enemies:
            enemy = Enemy(is_fast=True, direction=direction, location=location_objects[r][c])
            self.slow_enemies.append(enemy)
        
        player = data["player"]
        player_start = location_objects[player[0]][player[1]]
        self.player = Player(0, player_start)
        self.location_objects = location_objects
        self.draw_locations = locations
    
    def visualize(self):
        l = len(self.location_objects[0])
        for r in range(len(self.location_objects)):
            print("-" * (l*2))
            line = "|"
            for c in range(l):
                if self.location_objects[r][c] is None:
                    line += "X|"
                elif self.location_objects[r][c].is_goal:
                    line += "g|"
                else:
                    line += " |"
                    
            print(line)
        print("-" * (l*2))
    
    # returns list of valid actions 
    #
    # move: direction [0,1,2,3]
    # rotate: 4
    # wait: 5
    def get_valid_actions(self):
        location = self.player.location
        x = location.get_valid_directions()

        if location.gate is not None:
            x.append(4)

        x.append(5)
        return x

    def take_action(self, action):
        if action in [0,1,2,3]:
            return self.action_move(action)
        
        elif action == 4:
            return self.action_rotate()
        
        else:
            return self.action_wait()

    # Returns:
    #    -1 if you lost
    #    0 if the game is continuing
    #    1 if you won
    def action_move(self, direction):
        # fast enemies
        triggers = self.move_fast_enemies()
        if self.did_lose():
            return -1
        self.execute_triggers(triggers)

        # player
        triggers = []
        t = self.player.move(direction)
        if t is not None:
            triggers.append(t)
        if self.did_lose():
            return -1
        if self.did_win():
            return 1

        # all enemies
        triggers.extend(self.move_all_enemies())
        if self.did_lose():
            return -1
        self.execute_triggers(triggers)

        return 0

    # Returns:
    #    -1 if you lost
    #    0 if the game is continuing
    def action_wait(self):
        # fast enemies
        triggers = self.move_fast_enemies()
        if self.did_lose():
            return -1
        self.execute_triggers(triggers)

        # all enemies
        triggers.extend(self.move_all_enemies())
        if self.did_lose():
            return -1
        self.execute_triggers(triggers)
    
    # Rotation takes no time to do, so impossible to lose:
    #    Returns 0 if the game is continuing
    def action_rotate(self):
        self.player.location.gate.rotate()
        return 0
    

    # -------------------------------
    # Utility functions
    # -------------------------------
    def move_fast_enemies(self):
        triggers = []
        for enemy in self.fast_enemies:
            t = enemy.move()
            if t is not None:
                triggers.append(t)
        return triggers
    
    def move_all_enemies(self):
        triggers = []
        for enemy in self.fast_enemies:
            t = enemy.move()
            if t is not None:
                triggers.append(t)
        
        for enemy in self.slow_enemies:
            t = enemy.move()
            if t is not None:
                triggers.append(t)
        
        return triggers
    
    def execute_triggers(self, triggers):
        for trigger, times in triggers:
            trigger.rotate(times)

    def did_lose(self):
        for enemy in self.fast_enemies:
            if enemy.location == self.player.location:
                return True
        
        for enemy in self.slow_enemies:
            if enemy.location == self.player.location:
                return True

        return False

    def did_win(self):
        return self.player.location.is_goal



class Location:

    def __init__(self, gate=None, exit_trigger_map=None, is_goal=False, draw_loc=(0,0)):
        self.neighbors = [None, None, None, None]
        self.gate = gate
        self.exit_trigger_map = exit_trigger_map
        self.is_goal = is_goal
        self.draw_loc = draw_loc
    
    def get_valid_directions(self):
        valid = []

        for i in range(len(self.neighbors)):
            if self.neighbors[i] is None:
                continue

            dirs_blocked_by_gates = []
            if self.gate is not None:
                dirs_blocked_by_gates.extend(self.gate.directions_blocked)
            
            if self.neighbors[i].gate is not None:
                dirs_blocked_by_gates.extend(self.neighbors[i].gate.get_entering_blocked_dirs())

            if i not in dirs_blocked_by_gates:
                valid.append(i) 
        
        return valid

    def get_triggered(self, move_direction):
        if self.exit_trigger_map is None:
            return None
        
        if move_direction in self.exit_trigger_map:
            return self.exit_trigger_map[move_direction]
        
        return None
        


class Agent(object):

    def __init__(self, location):
        self.location = location
    

class Enemy(Agent):

    def __init__(self, is_fast, direction, location):
        super().__init__(location)
        self.direction = direction
        self.is_fast = is_fast
    
    # return the trigger that was triggered or None 
    def move(self):

        # enemies attempt to move
        # 1. forward
        # 2. left
        # 3. right
        # 4. backwards

        dirs_blocked = []
        if self.location.gate is not None:
            dirs_blocked.extend(self.location.gate.directions_blocked)

        for i in range(4):
            if self.location.neighbors[i] is not None:
                gate = self.location.neighbors[i].gate
                if gate is not None and i in gate.get_entering_blocked_dirs():
                    dirs_blocked.append(i)
        
        
        if self.location.neighbors[self.direction] is not None and self.direction not in dirs_blocked:
            triggered = self.location.get_triggered(self.direction)
            self.location = self.location.neighbors[self.direction]
            return triggered
        
        elif self.location.neighbors[(self.direction+3) % 4] is not None and ((self.direction+3) % 4) not in dirs_blocked:
            self.direction = (self.direction+3) % 4
            self.location = self.location.neighbors[self.direction]
        
        elif self.location.neighbors[(self.direction+1) % 4] is not None and ((self.direction+1) % 4) not in dirs_blocked:
            self.direction = (self.direction+1) % 4
            self.location = self.location.neighbors[self.direction]
        
        elif self.location.neighbors[(self.direction+2) % 4] is not None and ((self.direction+2) % 4) not in dirs_blocked:
            self.direction = (self.direction+2) % 4
            self.location = self.location.neighbors[self.direction]
                
        # only going forward can trigger a trigger
        return None


class Player(Agent):

    def __init__(self, direction, location):
        super().__init__(location)
        self.direction = direction

    # return the trigger that was triggered or None 
    def move(self, direction):
        if self.direction == direction:
            triggered = self.location.get_triggered(direction)
        else:
            triggered = None
            self.direction = direction
        self.location = self.location.neighbors[direction]
        return triggered



class Gate:

    def __init__(self, init_orientation):
        if init_orientation == 0:
            self.directions_blocked = [0,1]
        if init_orientation == 1:
            self.directions_blocked = [1,2]
        if init_orientation == 2:
            self.directions_blocked = [2,3]
        if init_orientation == 3:
            self.directions_blocked = [3,0]
    
    def get_entering_blocked_dirs(self):
        return [(self.directions_blocked[0]+2)%4, (self.directions_blocked[1]+2)%4]
    
    def rotate(self, times=1):
        self.directions_blocked = [(self.directions_blocked[0]+times)%4, (self.directions_blocked[1]+times)%4]


class Trigger:

    def __init__(self, trigger_map):
        self.trigger_map = trigger_map
    
    def agent_exit(self, agent):
        if agent.direction in self.trigger_map:
            gate, times = self.trigger_map[agent.direction]
            gate.rotate(times)

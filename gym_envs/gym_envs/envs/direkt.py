import gym
from gym import spaces
import numpy as np
import json
import pygame

class Direkt_v0(gym.Env):

    def __init__(self, render_mode=None, level=None):
        self.level = level
        self.observation_space = spaces.Box(0, 100, shape=(7,))
        self.action_space = spaces.Discrete(3)
        self.render_mode = render_mode
        self.window = None
        self.size = 512
        self.window_size = 512
        self.clock = None

        self.level = Level(level)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self._getobs({}), {}

    def _getobs(self, data):
        return np.array(data.decode().split(',')).astype(np.float32)

    def reward(self, obs):
        return -1

    def step(self, action):

        return {}, 5, False, False, {}

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
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # # First we draw the target
        # pygame.draw.rect(
        #     canvas,
        #     (255, 0, 0),
        #     pygame.Rect(
        #         pix_square_size * self._target_location,
        #         (pix_square_size, pix_square_size),
        #     ),
        # )
        # # Now we draw the agent
        # pygame.draw.circle(
        #     canvas,
        #     (0, 0, 255),
        #     (self._agent_location + 0.5) * pix_square_size,
        #     pix_square_size / 3,
        # )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

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
        self.fast_enemies = []
        self.slow_enemies = []
        self.player = None
        self.location_objects = None
        self.init_level(level_sheet)
    
    def init_level(self, level_sheet):
        f = open(level_sheet)
        data = json.load(f)

        locations = data["level_setup"]
        location_objects = [[None for i in range(len(locations[0]))] for j in range(len(locations))]
        for r in range(len(locations)):
            for c in range(len(locations[0])):
                if locations[r][c] == 0 or locations[r][c] == 1:
                    location_objects[r][c] = Location(is_goal=locations[r][c] == 1)
        
        for r in range(len(locations)):
            for c in range(len(locations[0])):
                # can have a north
                if r - 1 > 0:
                    location_objects[r][c].neighbors[0] = location_objects[r-1][c]
                
                # can have a west
                if c + 1 < len(locations[0]):
                    location_objects[r][c].neighbors[1] = location_objects[r][c + 1]
                
                # can have a south
                if r + 1 < len(locations):
                    location_objects[r][c].neighbors[2] = location_objects[r+1][c]
                
                # can have a west
                if c - 1 > 0:
                    location_objects[r][c].neighbors[1] = location_objects[r][c - 1]
        
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
            enemy = Enemy(direction, location_objects[r][c])
            self.slow_enemies.append(enemy)
        
        fast_enemies = data["fast_enemies"]
        for r, c, direction in fast_enemies:
            enemy = Enemy(is_fast=True, direction=direction, location=location_objects[r][c])
            self.slow_enemies.append(enemy)
        
        player = data["player"]
        self.player = Player(0, location_objects[0][1])
        self.location_objects = location_objects
    
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

        return location.get_valid_directions().extend([4,5])

    def take_action(self, action):
        if action in [0,1,2,3]:
            self.action_move(action)
        
        elif action == 4:
            self.action_rotate()
        
        else:
            self.action_wait()

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

    def __init__(self, north=None, west=None, south=None, east=None, gate=None, exit_trigger_map=None, is_goal=False):
        self.neighbors = [north, west, south, east]
        self.gate = gate
        self.exit_trigger_map = exit_trigger_map
        self.is_goal = is_goal
    
    def get_valid_directions(self):
        valid = []

        for i in range(len(self.neighbors)):
            if self.neighbors[i] is not None and i not in self.gate:
                valid.append(i)
        
        return i

    def get_triggered(self, move_direction):
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
        
        if self.location.neighbors[self.direction] is not None:
            triggered = self.location.get_triggered(self.direction)
            self.location = self.location.neighbors[self.direction]
            return triggered
        
        elif self.location.neighbors[self.direction+1 % 4] is not None:
            self.direction = self.direction+1 % 4
            self.location = self.location.neighbors[self.direction]
        
        elif self.location.neighbors[self.direction+3 % 4] is not None:
            self.direction = self.direction+3 % 4
            self.location = self.location.neighbors[self.direction]
        
        else:
            self.direction = self.direction+2 % 4
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

    def __init__(self, directions_blocked):
        self.directions_blocked = directions_blocked
    
    def rotate(self, times=1):
        self.directions_blocked = [self.directions_blocked[0]+times%4, self.directions_blocked[1]+times%4]


class Trigger:

    def __init__(self, trigger_map):
        self.trigger_map = trigger_map
    
    def agent_exit(self, agent):
        if agent.direction in self.trigger_map:
            gate, times = self.trigger_map[agent.direction]
            gate.rotate(times)
    


d = Direkt_v0(render_mode="human", level="levels/level1.json")
while True:
    d.render()
# l = Level("levels/level1.json")
# l.visualize()

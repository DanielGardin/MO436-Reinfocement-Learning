from pacman.utils import discrete, manhattan_distance
from pacman.actions import Actions
from pacman import agents
import time, signal, os
import numpy as np
from typing import Tuple, List, Any

import matplotlib as mlp
import matplotlib.pyplot as plt
import seaborn as sns
import pygame

def is_running_in_jupyter():
    try:
        get_ipython() # type: ignore
        return True
    except NameError:
        return False

if is_running_in_jupyter():
    from IPython.display import clear_output


def chaikins_corner_cutting(coords, refinements=5):
    coords = np.array(coords)

    for _ in range(refinements):
        L = coords.repeat(2, axis=0)
        R = np.empty_like(L)
        R[0] = L[0]
        R[2::2] = L[1:-1:2]
        R[1:-1:2] = L[2::2]
        R[-1] = L[-1]
        coords = L * 0.75 + R * 0.25

    return coords

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def load_ghost(ghost_name):

    if hasattr(agents, ghost_name):
        return getattr(agents, ghost_name)

    raise Exception(f'The ghost {ghost_name} is not found')

class timeout:
    def __init__(self, seconds=0., error_message='Execution timed out'):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        if os.name == 'nt': return

        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, type, value, traceback):
        if os.name == 'nt': return
        signal.alarm(0)


class PacmanEnv:
    PACMAN_SPEED  = 1
    SCARED_TIME   = 40
    COLLISION_TOL = .7
    MAX_STEPS     = 100

    FOOD_REWARD  = 1
    GHOST_REWARD = 10
    WIN_REWARD   = 50

    TIME_PENALTY = 1
    LOSE_PENALTY = 20
    WALL_PENALTY = 1

    PACMAN_COLOR = style.YELLOW
    GHOST_COLORS = [
        style.RED,
        style.CYAN,
        style.MAGENTA,
        style.GREEN
    ]
    SCARED_GHOST_COLOR = style.BLUE

    GHOST_HEXCOLORS = [
        (252, 0., 0.),
        (0, 255, 255),
        (252, 180, 255),
        (252, 180/155, 85/155)
    ]

    SCARED_GHOST_HEXCOLOR = (36, 36, 255)

    RAW_GHOST_SHAPE = [
        ( 0,    0.3 ),
        ( 0.25, 0.75 ),
        ( 0.5,  0.3 ),
        ( 0.75, 0.75 ),
        ( 0.75, -0.5 ),
        ( 0.5,  -0.75 ),
        (-0.5,  -0.75 ),
        (-0.75, -0.5 ),
        (-0.75, 0.75 ),
        (-0.5,  0.3 ),
        (-0.25, 0.75 )
    ]

    GHOST_SHAPE = chaikins_corner_cutting(RAW_GHOST_SHAPE, 1)

    GRID_SIZE = 50

    def __init__(self,
                 layout_text,
                 ghost_names=None,
                 render_mode = 'ansi',
                 state_space = "default",
                 config = None
                 ):
        self.width  = len(layout_text[0])
        self.height = len(layout_text)
        self.layout = layout_text

        self.set_render(render_mode)
        self.state_space = state_space
        self.state_dims  = (2,)

        self.walls    = np.zeros((self.width, self.height))
        self.env_food = np.zeros((self.width, self.height))

        self._capsules    = set()
        self.ghosts      = []
        self.ghost_names = ghost_names

        self.initial_position  = (-1, -1)
        
        self._process_layout(layout_text, ghost_names)

        self.ready = False

        if config is None: return
        for configuration, value in config.items():
            if hasattr(self.__class__, configuration):
                setattr(self, configuration, value)
            
            else:
                raise ValueError(f"No configuration named {config}.")


    def _process_layout(self, layout_text, ghost_names=None):
        """
        Coordinates are flipped from the input format to the (x,y) convention here

        The shape of the maze.  Each character
        represents a different type of object.
        - % - Wall
        - . - Food
        - o - Capsule
        - n - Ghost, where n is a numeric value, or G, if all ghosts are the same
        - P - Pacman
        Other characters are ignored.
        """
        idx = 0
        max_y = self.height - 1
        for y in range(self.height):
            for x in range(self.width):
                layout_char = layout_text[max_y - y][x]

                if layout_char == '%':
                    self.walls[x, y] = 1
                elif layout_char == '.':
                    self.env_food[x, y] = 1
                elif layout_char == 'o':
                    self._capsules.add((x, y))
                elif layout_char == 'P':
                    self.initial_position = (x, y)
                elif layout_char.isnumeric():
                    if ghost_names is None:
                        ghost = agents.RandomGhost((x, y))
                    else:
                        idx = int(layout_char)
                        ghost = load_ghost(ghost_names[idx])((x, y))
                    
                    self.ghosts.append(ghost)
    
                elif layout_char == 'G':
                    if ghost_names is None:
                        ghost = agents.RandomGhost((x, y))
                    elif isinstance(ghost_names, str):
                        ghost = load_ghost(ghost_names)((x, y))
                    else:
                        ghost = load_ghost(ghost_names[idx])((x, y))
                        idx += 1

                    self.ghosts.append(ghost)


    @classmethod
    def from_file(cls, 
                  name:str,
                  ghost_names=None,
                  render_mode = 'ansi',
                  state_space = "default",
                  config = None
                  ):
        """
        Loads a layout from a ./layouts folder. The layout name must be
        provided without extension.
        """
        import os

        path = os.path.dirname(__file__)
        path = os.path.join(path, f'layouts/{name}.lay')

        with open(os.path.realpath(path), 'r') as file:
            layout_text = [line.strip() for line in file]

        return cls(layout_text, ghost_names, render_mode, state_space, config)


    @classmethod
    def contourDanger(cls, env_side=7, initial_pos = None, ghost_pos=(2,2), ghost_names = None, render_mode='ansi', state_space='default', config=None):
        if env_side < 4:
            raise ValueError("Environment cannot be smaller than 4 in each side.")

        border = ["%"] * (env_side + 2)
        inside = ['%'] + [' '] * env_side + ['%']

        env = [border] + [inside.copy() for _ in range(env_side)] + [border]

        if initial_pos is None:
            initial_pos = (env_side//2 + 1, env_side//2 + 1)

        x_agent, y_agent = initial_pos
        x_ghost, y_ghost = ghost_pos

        env[y_agent][x_agent] = 'P'
        env[y_ghost][x_ghost] = 'G'

        env[1][1] = '.'

        layout = [''.join(line) for line in env]

        return cls(layout, ghost_names, render_mode, state_space, config)

    
    @classmethod
    def classic(cls, size="small", ghost_names = None, render_mode='ansi', state_space='default', config=None):
        env_name = f"{size}Classic"

        return cls.from_file(env_name, ghost_names, render_mode, state_space, config)


    def reset(self, *, random_init=False, seed=None) -> Tuple[Any, bool]:
        """
        Resets the environment to its initial state.
        """
        self.ready = True

        self.direction        = Actions.NOOP
        self.current_food     = self.env_food.copy()
        self.current_capsules = self._capsules.copy()
        for ghost in self.ghosts: ghost.reset()

        self.score        = 0
        self.current_step = 0

        if random_init:
            self.position = self.get_random_legal_position()
        
        else:
            self.position = self.initial_position

        self.lose  = False
        done = self.is_terminal()
        
        self.render()

        return self.observation(), done


    def __repr__(self) -> str:
        representation = ""

        if self.ready:

            info = self.observation('info')

            for feature, value in info.items():
                representation += f'{feature:<20} {value}\n'

        else:
            representation = "Not ready Pacman environment."

        return representation


    def get_random_legal_position(self) -> Tuple[int, int]:
        """
        Returns a non-wall position in the grid, which can be used to
        initialize the state from a random start.
        """
        free_spaces = self.get_all_free_positions()

        idx = np.random.randint(len(free_spaces))

        return free_spaces[idx]
    
    def get_all_free_positions(self):
        return list(zip(*np.where(self.walls == 0)))
    
    def get_legal_actions(self, position) -> List[str]:
        """
        Returns all actions that can be executed in a given position in
        the current state.
        """
        legal_actions = []
        x    , y     = position
        x_int, y_int = discrete(position)

        possible_actions = []

        if x_int != x:
            possible_actions = [Actions.LEFT, Actions.RIGHT]
        
        elif y_int != y:
            possible_actions = [Actions.UP, Actions.DOWN]
        
        else:
            possible_actions = Actions.actions

        for action in possible_actions:
            dx, dy = Actions.action_to_vector(action)
            new_pos = discrete((x_int + dx, y_int + dy))

            if not self.haswall(new_pos):
                legal_actions.append(action)
        
        return legal_actions


    def __str__(self) -> str:
        """Ansi representation of a state."""

        grid = [[' ' for _ in range(self.height)] for _ in range(self.width)]
        
        np.chararray((self.width, self.height), itemsize=1)

        for y in range(self.height):
            for x in range(self.width):
                if self.haswall((x, y)):
                    grid[x][y] = '%'
                
                elif self.hasfood((x, y)):
                    grid[x][y] = '.'
                
                elif self.hascapsule((x, y)):
                    grid[x][y] = 'o'


        x, y = discrete(self.position)
        grid[x][y] = self.PACMAN_COLOR

        if self.direction == Actions.UP:
            grid[x][y] += '^'
        
        elif self.direction == Actions.DOWN:
            grid[x][y] += 'v'

        elif self.direction == Actions.LEFT:
            grid[x][y] += '<'
        
        elif self.direction == Actions.RIGHT:
            grid[x][y] += '>'
        
        else:
            grid[x][y] += 'P'
        
        grid[x][y] += style.RESET

        for i, ghost in enumerate(self.ghosts):
            x, y = discrete(ghost.position)

            if ghost.is_scared():
                grid[x][y] = self.SCARED_GHOST_COLOR
            
            else:
                grid[x][y] = self.GHOST_COLORS[i % 4]


            if ghost.direction == Actions.UP:
                grid[x][y] += 'M'
            
            elif ghost.direction == Actions.DOWN:
                grid[x][y] += 'W'

            elif ghost.direction == Actions.LEFT:
                grid[x][y] += 'E'
            
            elif ghost.direction == Actions.RIGHT:
                grid[x][y] += '3'
            
            else:
                grid[x][y] += 'G'
            
            grid[x][y] += style.RESET


        grid = reversed([list(i) for i in zip(*grid)])

        grid_text  = '\n'.join([' '.join(row) for row in grid])
        score_text = f"\nScore: {self.score}"

        terminal_text = ' Game Over!' if self.is_lose() else ''

        return grid_text + score_text + terminal_text


    ##################################################################
    # The following methods are used for gather specific information #
    # about the state in order to build an observation to the actor. #
    ##################################################################

    def get_score(self) -> int : return self.score

    def get_ghosts_position(self) -> List[Tuple[int, int]]:
        return [ghost.position for ghost in self.ghosts]

    def get_position(self): return self.position

    def haswall(self, position) -> bool: return bool(self.walls[position])

    def hasfood(self, position) -> bool: return bool(self.current_food[position])

    def hascapsule(self, position) -> bool: return position in self.current_capsules

    def get_food_positions(self): return list(zip(*np.where(self.current_food == 1)))

    def get_num_food(self) -> int: return np.count_nonzero(self.current_food)

    def get_total_food(self) -> int: return np.count_nonzero(self.env_food)

    def is_terminal(self) -> bool: return self.is_win() or self.is_lose()
    
    def is_win(self) -> bool:  return not self.is_lose() and self.get_num_food() == 0

    def is_lose(self) -> bool: return self.current_step > self.MAX_STEPS or self.lose

    def search_dist(self, source:Tuple[int, int], target) -> int:
        if isinstance(target, tuple):
            target = [target]

        if len(target) == 0:
            return -1

        visited = {}
        distance = int(0)
    
        queue = [(source, distance)]

        while queue:
            candidate, distance = queue.pop(0)

            if candidate in visited:
                continue

            if self.haswall(candidate):
                continue

            if candidate in target:
                return distance
            
            x, y = candidate
            new_candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

            for new_candidate in new_candidates:
                queue.append((new_candidate, distance + 1))
            
            visited[candidate] = distance

        return -1

    def search_radar(self, source, target, cap=2):
        if isinstance(target, tuple):
            target = [target]

        if len(target) == 0:
            return 0

        visited = {}
        distance = int(0)
    
        queue = [(source, distance)]

        found = 0

        while queue:
            candidate, distance = queue.pop(0)

            if distance > cap:
                return found

            if candidate in visited:
                continue

            if self.haswall(candidate):
                continue

            if candidate in target:
                found += 1
            
            x, y = candidate
            new_candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

            for new_candidate in new_candidates:
                queue.append((new_candidate, distance + 1))
            
            visited[candidate] = distance


    def linear_radar(self, position, direction):
        if self.haswall(discrete(position)):
            return 0

        next_position = Actions.apply_action(position, direction)

        return self.linear_radar(next_position, direction) + 1


    ##################################################################
    # Environment mechanism, this implementation follows Gymnasium   #
    # convention for environment interaction, with step and reset.   #
    ##################################################################

    def check_collision(self, ghost) -> bool:
        return manhattan_distance(self.position, ghost.position) < self.COLLISION_TOL


    def step(self, action) -> Tuple[Any, int, bool, Any]:
        """
        Executes a provided action to the current state and returns a tuple
        containing

        observation : The observation after the action execution
        reward      : Reward obtained after executing the action
        done        : Whether the state is terminal (or truncated) or not
        """
        if not self.ready:
            raise Exception("Environment not reseted before step is called. Make sure to reset the environment after instanciated.")

        if self.is_terminal():
            self.render()

            return self.observation(), 0, True, self.observation('info')

        self.current_step += 1
        score_change = - self.TIME_PENALTY

        # Every agent decides an action
        ghost_actions = {ghost : ghost.act(self) for ghost in self.ghosts}

        legal_actions = self.get_legal_actions(self.position)

        if action in legal_actions:
            x, y = self.position
            dx, dy = Actions.action_to_vector(action, self.PACMAN_SPEED)

            self.position = (x + dx, y + dy)
        
        else:
            score_change -= self.WALL_PENALTY

        self.direction = action

        for ghost, ghost_action in ghost_actions.items():
            if Actions.reverse_direction(ghost_action) == self.direction and self.check_collision(ghost):
                if ghost.is_scared():
                    score_change += self.GHOST_REWARD
                    ghost.reset()
                    continue

                else:
                    score_change -= self.LOSE_PENALTY
                    self.lose = True
                    break

            ghost.apply_action(self, ghost_action)

            if self.check_collision(ghost):
                if ghost.is_scared():
                    score_change += self.GHOST_REWARD
                    ghost.reset()

                else:
                    score_change -= self.LOSE_PENALTY
                    self.lose = True
                    break


        discrete_pos  = discrete(self.position)
        if manhattan_distance(self.position, discrete_pos) <= 0.5:
            if self.hasfood(discrete_pos):
                score_change += self.FOOD_REWARD
                self.current_food[discrete_pos] = 0

            if self.hascapsule(discrete_pos):
                for ghost in self.ghosts: ghost.scare(self.SCARED_TIME)
                self.current_capsules.remove(discrete_pos)

        if not self.is_lose() and self.get_num_food() == 0:
            score_change += self.WIN_REWARD

        self.score += score_change

        self.render()

        return self.observation(), score_change, self.is_terminal(), self.observation('info')


    def set_render(self, render_mode):
        self.render_mode = render_mode

        if render_mode == 'human':
            self.screen = None

        return self


    def render(self):
        if self.render_mode == 'ansi':
            if is_running_in_jupyter(): clear_output()
            print(self)

        elif self.render_mode == 'human':
            if self.screen is None:
                self.screen = pygame.display.set_mode((self.width * self.GRID_SIZE, self.height * self.GRID_SIZE))
                pygame.display.set_caption("Pacman environment")

            pygame.event.pump()

            self._render_pygame()

    
    def _render_pygame(self):
        self.screen.fill((0, 0, 0))

        for x in range(self.width):
            for y in range(self.height):
                if self.haswall((x,y)):
                    pygame.draw.rect(self.screen, (36, 36, 255), (x*self.GRID_SIZE, (self.height-y-1)*self.GRID_SIZE, self.GRID_SIZE, self.GRID_SIZE))
                

                if self.hasfood((x,y)):
                    pygame.draw.circle(self.screen, (255, 255, 255), ((x+0.5)* self.GRID_SIZE, (self.height-y-0.5)* self.GRID_SIZE), self.GRID_SIZE/20)

        for (x, y) in self.current_capsules:
            pygame.draw.circle(self.screen, (255, 255, 255), ((x+0.5)* self.GRID_SIZE, (self.height-y-0.5)* self.GRID_SIZE), self.GRID_SIZE/5)

        x_pacman, y_pacman = self.position

        y_pacman = (self.height - 0.5 - y_pacman) * self.GRID_SIZE
        x_pacman = (x_pacman + 0.5) * self.GRID_SIZE

        radius = 0.4 * self.GRID_SIZE

        pygame.draw.circle(self.screen, (252, 252, 0), (x_pacman, y_pacman), radius)

        dx, dy = (1, 0) if self.direction == Actions.NOOP else Actions.action_to_vector(self.direction) 

        mouth = [
            (x_pacman + (dx + abs(dx) - 1) * radius, y_pacman + (-dy + abs(dy) - 1) * radius),
            (x_pacman, y_pacman),
            (x_pacman + (dx - abs(dx) + 1) * radius, y_pacman + (-dy - abs(dy) + 1) * radius)
        ]

        pygame.draw.polygon(self.screen, (0, 0, 0), mouth)

        pupil_shift = 0.05 * self.GRID_SIZE

        eyes = {
            Actions.NOOP : (0, 0),
            Actions.UP : (0, -pupil_shift),
            Actions.DOWN : (0, pupil_shift),
            Actions.LEFT : (-pupil_shift, 0),
            Actions.RIGHT : (pupil_shift, 0)
        }

        for i, ghost in enumerate(self.ghosts):
            x_ghost, y_ghost = ghost.position

            y_ghost = (self.height - 0.5 - y_ghost) * self.GRID_SIZE
            x_ghost = (x_ghost + 0.5) * self.GRID_SIZE


            coords = [(i * 0.5 * self.GRID_SIZE + x_ghost, j * 0.5 * self.GRID_SIZE + y_ghost) for (i, j) in self.GHOST_SHAPE]

            color = self.SCARED_GHOST_HEXCOLOR if ghost.is_scared() else self.GHOST_HEXCOLORS[i]

            pygame.draw.polygon(self.screen, color, coords)


            dx, dy = eyes[ghost.direction]

            eye_offset = 0.05 * self.GRID_SIZE
            eye_shape = (0.2 * self.GRID_SIZE, 0.3 * self.GRID_SIZE)

            pupil_radius = 0.09 * self.GRID_SIZE

            pygame.draw.ellipse(self.screen, (255, 255, 255), (x_ghost - (eye_shape[0] + eye_offset), y_ghost - eye_shape[1] , *eye_shape))
            pygame.draw.ellipse(self.screen, (255, 255, 255), (x_ghost + eye_offset, y_ghost - eye_shape[1] , *eye_shape))

            pygame.draw.circle(self.screen, (0, 0, 0), (x_ghost - (eye_shape[0]//2 + eye_offset - dx), y_ghost - eye_shape[1] // 2 + dy), pupil_radius)
            pygame.draw.circle(self.screen, (0, 0, 0), (x_ghost + eye_offset + eye_shape[0]//2 + dx, y_ghost - eye_shape[1] // 2 + dy), pupil_radius)

        pygame.display.update()        


    def close(self):
        if self.render_mode == "human":
            pygame.quit()
            self.screen = None


    def observation(self, space=None) -> Any:
        """
        Provides an observation of the current state to the agent.
        This might be changed accordingly to the intended information
        the agent uses to decide.
        """
        if not self.ready: raise Exception("Environment not initializated before a observation was requested. Make sure to reset the environment after instanciated.")
        if space is None: space = self.state_space

        raw_representation = {
            'position'       : self.position,
            'collected food' : tuple(1 - self.current_food[self.env_food == 1])
        }

        if self.ghosts:
            raw_representation['ghost positions'] = tuple(ghost.position  for ghost in self.ghosts)
            raw_representation['ghost direction'] = tuple(ghost.direction for ghost in self.ghosts)
            raw_representation['scared']   = tuple(int(ghost.is_scared()) for ghost in self.ghosts)
        
        if self._capsules:
            raw_representation['collected capsules'] = tuple(int(capsule not in self.current_capsules) for capsule in self._capsules)

        if space == 'info':
            return raw_representation
    
        elif space == 'default':
            return tuple(raw_representation.values())

        elif space == 'features':
            features = []

            for action in Actions.actions:
                new_pos = discrete(Actions.apply_action(self.position, action))
                if self.haswall(new_pos):
                    new_pos = self.position

                action_features = { # 
                    "distance to ghost"   : self.search_dist(new_pos, self.get_ghosts_position()) / self.width,
                    "distance to food"    : self.search_dist(new_pos, self.get_food_positions()) / self.width,
                    "active ghost 1 step" : self.search_radar(new_pos, self.get_ghosts_position(), cap=1),
                    "active ghost 2 step" : self.search_radar(new_pos, self.get_ghosts_position(), cap=2),
                    "has food"            : int(self.hasfood(new_pos)),
                }

                features.append(tuple(action_features.values()))
            return tuple(features)
        
        elif space == 'pixel':

            state = np.zeros((3, self.height, self.width))

            for x in range(self.width):
                for y in range(self.height):
                    if self.haswall((x, y)):
                        state[:, y, x] = (0., 0., 1.)
                    
                    elif self.hasfood((x, y)):
                        state[:, y, x] = (1., 1., 1.)
                    
                    else:
                        state[:, y, x] = (0., 0., 0.)
            
            x_pacman, y_pacman = discrete(self.position)

            state[:, y_pacman, x_pacman] = (1., 1., 0.)

            for ghost in self.ghosts:
                x, y = discrete(ghost.position)

                if ghost.is_scared():
                    state[:, y, x] = (0, 1., 0.)
                
                else:
                    state[:, y, x] = (1., 0., 0.)


                if ghost.direction == Actions.UP:
                    state[-1, y, x] += .2
                
                elif ghost.direction == Actions.DOWN:
                    state[-1, y, x] += .4

                elif ghost.direction == Actions.LEFT:
                    state[-1, y, x] += .6
                
                elif ghost.direction == Actions.RIGHT:
                    state[-1, y, x] += .8

            for (x, y) in self.current_capsules:
                state[:, y, x] = (.2, .2, .2)

            return tuple(state.reshape((-1,)))


    def run_policy(self,
                   policy,
                   timeout_time = 0,
                   delay        = 0.,
                   seed         = None):
        """
        Run an entire game given a policy.

        # Arguments

        policy : Agent
            the policy being evaluated. Must inherit from Agent class
        
        timeout : int, default=0
            computational limit, in seconds, for an agent to output an action.
            Letting it to be 0 means no timeout.
        
        delay : int, default=0set
            time, in seconds, between steps.
        """
        obs, done = self.reset(seed=seed)

        experiences = []

        while not done:
            time.sleep(delay)

            with timeout(timeout_time):
                action = policy.act(obs)

            next_obs, reward, done, info = self.step(action)

            experiences.append((obs, action, reward, next_obs))

            obs = next_obs

        self.close()

        return experiences


    def check_policy(self, policy):
        self.fig, self.ax = plt.subplots(figsize=(self.width + 1, self.height))
        self.ax.set_axis_off()

        if self.state_space == "default":
            info = self.observation('info')

            Qmap = np.zeros_like(self.walls)

            for coord in self.get_all_free_positions():
                for action in Actions.actions:
                    new_pos = Actions.calculate_next_position(coord, action)
                    
                    info['position'] = coord

                    state = tuple(info.values())
            
                    Qmap[new_pos] += policy.Q(state, action) / 4



            sns.heatmap(Qmap.T, alpha=.5, vmax=50, vmin=-50)

            print(Qmap)
    
        for x in range(self.width):
            for y in range(self.height):

                if self.haswall((x,y)):
                    rect = plt.Rectangle((x, y), 1, 1, color='blue')
            
                
                self.ax.add_patch(rect)

                if self.hasfood((x,y)):
                    food = plt.Circle((x + 1/2, y + 1/2), 0.05, color='white')
                    
                    self.ax.add_patch(food)


        x_pacman, y_pacman = self.position

        angle_conversion = {
            Actions.NOOP  : 0,
            Actions.RIGHT : 0,
            Actions.UP    : 90,
            Actions.LEFT  : 180,
            Actions.DOWN  : 270
        }

        angle = angle_conversion[self.direction]

        pacman = mlp.patches.Wedge((x_pacman + 1/2, y_pacman + 1/2), 0.38, angle + 35, angle - 35, width= 0.38, color='yellow')

        self.ax.add_patch(pacman)

        eyes = {
            Actions.NOOP : (0, 0),
            Actions.UP : (0, -0.2),
            Actions.DOWN : (0, 0.2),
            Actions.LEFT : (-0.2, 0),
            Actions.RIGHT : (0.2, 0)
        }

        for i, ghost in enumerate(self.ghosts):
            x_ghost, y_ghost = ghost.position

            coords = [(-i * 0.5 + x_ghost + 1/2, -j * 0.5 + y_ghost + 1/2) for (i, j) in self.GHOST_SHAPE]

            color = self.SCARED_GHOST_HEXCOLOR if ghost.is_scared() else self.GHOST_HEXCOLORS[i]

            ghost_body = mlp.patches.Polygon(coords, color=tuple(c/255 for c in color))

            dx, dy = eyes[ghost.direction]

            left_eye  = mlp.patches.Ellipse((x_ghost + 1/2 + 0.5 * (-0.3 + dx)/1.5, y_ghost + 1/2 + 0.5 * (0.3 - dy)/1.5), 0.3*0.5, 0.4*0.5, color="white")
            right_eye = mlp.patches.Ellipse((x_ghost + 1/2 + 0.5 * (0.3 + dx)/1.5, y_ghost + 1/2 + 0.5 * (0.3 - dy)/1.5), 0.3*0.5, 0.4*0.5, color="white")

            left_pupil  = mlp.patches.Circle((x_ghost + 1/2 + 0.5 * (-0.3 + dx)/1.5, y_ghost + 1/2 + 0.5 * (0.3 - dy)/1.5), 0.1*0.5, color="black")
            right_pupil = mlp.patches.Circle((x_ghost + 1/2 + 0.5 * (0.3 + dx)/1.5, y_ghost + 1/2 + 0.5 * (0.3 - dy)/1.5), 0.1*0.5, color="black")

            self.ax.add_patch(ghost_body)
            self.ax.add_patch(left_eye)
            self.ax.add_patch(right_eye)

            self.ax.add_patch(left_pupil)
            self.ax.add_patch(right_pupil)


        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)

                
        plt.show()
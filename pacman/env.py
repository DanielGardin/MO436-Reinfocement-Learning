from pacman.utils import discrete, manhattan_distance
from collections.abc import Sequence 
from pacman.actions import Actions
from pacman import agents
import time, signal, os
import numpy as np

def is_running_in_jupyter():
    try:
        get_ipython()
        return True
    except NameError:
        return False

if is_running_in_jupyter():
    from IPython.display import clear_output

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

    TIME_PENALTY = 0
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

        self.render_mode = render_mode
        self.state_space = state_space

        self.walls    = np.zeros((self.width, self.height))
        self.env_food = np.zeros((self.width, self.height))

        self._capsules    = set()
        self.ghosts      = []
        self.ghost_names = ghost_names

        self.initial_position  = (None, None)
        
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
        - \# - Ghost, where # is a numeric value, or G, if all ghosts are the same
        - P - Pacman
        Other characters are ignored.
        """
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
                    else:
                        ghost = load_ghost(ghost_names)((x, y))

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


    def reset(self, *, random_init=False, seed=None):
        """
        Resets the environment to its initial state.
        """

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


        done = self.is_terminal()

        self.ready = True

        self.render()

        return self.observation(), done


    def __repr__(self):
        representation = ""

        if self.ready:

            info = self.observation('info')

            for feature, value in info.items():
                representation += f'{feature:<20} {value}\n'

        else:
            representation = "Not ready Pacman environment."

        return representation


    def set_render(self, render_mode):
        self.render_mode = render_mode


    def get_random_legal_position(self):
        """
        Returns a non-wall position in the grid, which can be used to
        initialize the state from a random start.
        """
        free_spaces = np.where(self.walls == 0)
        n_spaces    = len(free_spaces[0])

        idx = np.random.randint(n_spaces)

        return (free_spaces[0][idx], free_spaces[1][idx])
    
    
    def get_legal_actions(self, position):
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


    def __str__(self):
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

    def get_score(self): return self.score

    def getghosts_position(self):
        return [ghost.position for ghost in self.ghosts]

    def get_position(self): return self.position

    def haswall(self, position): return bool(self.walls[position])

    def hasfood(self, position): return bool(self.current_food[position])

    def hascapsule(self, position): return position in self.current_capsules

    def get_num_food(self): return np.count_nonzero(self.current_food)

    def get_total_food(self) : return np.count_nonzero(self.env_food)

    def is_terminal(self): return self.is_win() or self.is_lose()
    
    def is_win(self):  return self.get_num_food() == 0

    def is_lose(self): return self.current_step > self.MAX_STEPS or any([manhattan_distance(self.position, ghost.position) < self.COLLISION_TOL and not ghost.is_scared() for ghost in self.ghosts])

    def search_dist(self, source, target):
        if isinstance(target, tuple):
            target = [target]

        visited = {}
        queue = [source]
        dists = [0]

        while queue:
            candidate = queue.pop(0)
            distance  = dists.pop(0)

            if candidate in visited:
                continue

            if self.haswall(candidate):
                continue

            if candidate in target:
                break
            
            x, y = candidate
            new_candidates = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]

            for new_candidate in new_candidates:
                queue.append(new_candidate)
                dists.append(distance + 1)
            
            visited[candidate] = distance
            
        return distance

    ##################################################################
    # Environment mechanism, this implementation follows Gymnasium   #
    # convention for environment interaction, with step and reset.   #
    ##################################################################

    def resolve_collision(self, ghost):
        score_change = 0

        if manhattan_distance(self.position, ghost.position) < self.COLLISION_TOL \
           and ghost.direction != self.position:
            if ghost.is_scared():
                score_change += self.GHOST_REWARD
                ghost.reset()

            else:
                score_change -= self.LOSE_PENALTY

        return score_change


    def step(self, action):
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

            return self.observation(), 0, True

        self.current_step += 1
        score_change = 0

        # Every agent decides an action
        ghost_actions = {ghost : ghost.act(self) for ghost in self.ghosts}

        print(ghost_actions)

        legal_actions = self.get_legal_actions(self.position)

        if action in legal_actions:
            x, y = self.position
            dx, dy = Actions.action_to_vector(action, self.PACMAN_SPEED)

            self.position = (x + dx, y + dy)
        
        else:
            score_change -= self.WALL_PENALTY

        for ghost, ghost_action in ghost_actions.items():
            score_change += self.resolve_collision(ghost)

            ghost.apply_action(self, ghost_action)

            score_change += self.resolve_collision(ghost)


        discrete_pos  = discrete(self.position)
        self.direction = action

        score_change -= self.TIME_PENALTY
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


    def render(self):
        if self.render_mode == 'ansi':
            if is_running_in_jupyter(): clear_output()
            print(self)

        

    def observation(self, space=None):
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
            raw_representation['ghost positions'] = tuple(ghost.position for ghost in self.ghosts)
            raw_representation['is scared'] = tuple(int(ghost.is_scared()) for ghost in self.ghosts)
        
        if self._capsules:
            raw_representation['collected capsules'] = tuple(int(capsule not in self.current_capsules) for capsule in self._capsules)

        if space == 'info':
            return raw_representation
    
        elif space == 'default':
            return tuple(raw_representation.values())
            

        

    def run_policy(self,
                   policy : agents.Agent,
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
        
        delay : int, default=0
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

        return experiences

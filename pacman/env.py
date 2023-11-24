from pacman.utils import discrete, manhattan_distance
from pacman.actions import Actions
from pacman import agents
import time, signal, sys
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
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


class PacmanEnv:
    PACMAN_SPEED  = 1
    SCARED_TIME   = 40
    COLLISION_TOL = .7

    FOOD_REWARD  = 10
    GHOST_REWARD = 200
    WIN_REWARD   = 500
    TIME_PENALTY = 1
    LOSE_PENALTY = 200

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
                 render_mode = 'ansi'
                 ):
        self.width  = len(layout_text[0])
        self.height = len(layout_text)

        self.walls    = np.zeros((self.width, self.height))
        self.food     = np.zeros((self.width, self.height))
        self.capsules = set()

        self.position  = (None, None)
        self.direction = Actions.NOOP
        self.ghosts    = []

        self.layout      = layout_text
        self.ghost_names = ghost_names
        self._process_layout(layout_text, ghost_names)  

        self.score        = 0
        self.done         = False
        self._win         = False
        self._lose        = False

        self.render_mode = render_mode
    
    def __hash__(self):
        state_repr = tuple(zip(*np.where(self.food == 1))) + \
                     tuple(self.capsules) + \
                     tuple(ghost.position for ghost in self.ghosts) + \
                     self.position

        return hash(state_repr)


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
                    self.food[x, y] = 1
                elif layout_char == 'o':
                    self.capsules.add((x, y))
                elif layout_char == 'P':
                    self.position = (x, y)
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
    def from_file(cls, name:str, **kwargs):
        """
        Loads a layout from a ./layouts folder. The layout name must be
        provided without extension.
        """
        import os

        path = os.path.dirname(__file__)
        path = os.path.join(path, f'layouts/{name}.lay')

        with open(os.path.realpath(path), 'r') as file:
            layout_text = [line.strip() for line in file]

        return cls(layout_text, **kwargs)


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

        x_int, y_int = discrete(position)

        for action in Actions.actions:
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

        terminal_text = ' Game Over!' if self.islose() else ''

        return grid_text + score_text + terminal_text


    ##################################################################
    # The following methods are used for gather specific information #
    # about the state in order to build an observation to the actor. #
    ##################################################################

    def get_score(self): return self.score

    def get_ghosts_position(self):
        return [ghost.position for ghost in self.ghosts]

    def get_position(self): return self.position

    def haswall(self, position): return bool(self.walls[position])

    def hasfood(self, position): return bool(self.food[position])

    def hascapsule(self, position): return position in self.capsules

    def get_num_food(self): return self.food.sum()

    def isterminal(self): return self.done
    
    def iswin(self):  return self._win

    def islose(self): return self._lose

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
                self._lose = True
                self.done  = True

        return score_change


    def reset(self, *, random_init=False, seed=None):
        """
        Resets the environment to its initial state.
        """
        self.done  = False
        self._win  = False
        self._lose = False

        self.direction = Actions.NOOP
        self.ghosts   = []

        self._process_layout(self.layout, ghost_names=self.ghost_names)

        self.score        = 0
        self.position     = self.get_random_legal_position()

        self.render()


    def step(self, action):
        """
        Executes a provided action to the current state and returns a tuple
        containing

        observation : The observation after the action execution
        reward      : Reward obtained after executing the action
        done        : Whether the state is terminal (or truncated) or not
        """
        if self.done:
            self.render()

            return self.observation(), 0, True

        score_change = 0

        # Every agent decides an action
        ghost_actions = [Actions.NOOP] * len(self.ghosts)

        for i, ghost in enumerate(self.ghosts):
            ghost_actions[i] = ghost.act(self)

        # Resolve collisions
        legal_actions = self.get_legal_actions(self.position)

        if action in legal_actions:
            x, y = self.position
            dx, dy = Actions.action_to_vector(action, self.PACMAN_SPEED)

            self.position = (x + dx, y + dy)

        for ghost, ghost_action in zip(self.ghosts, ghost_actions):
            score_change += self.resolve_collision(ghost)

            ghost.apply_action(self, ghost_action)

            score_change += self.resolve_collision(ghost)
        

        discrete_pos  = discrete(self.position)
        self.direction = action

        score_change -= self.TIME_PENALTY
        if manhattan_distance(self.position, discrete_pos) <= 0.5:
            if self.hasfood(discrete_pos):
                score_change += self.FOOD_REWARD
                self.food[discrete_pos] = 0

            if self.hascapsule(discrete_pos):
                for ghost in self.ghosts: ghost.scare(self.SCARED_TIME)
                self.capsules.remove(discrete_pos)

        if self.get_num_food() == 0 and not self.done:
            score_change += self.WIN_REWARD
            self._win = True
            self.done = True

        self.score += score_change

        self.render()

        return self.observation(), score_change, self.done


    def render(self):
        if self.render_mode == 'ansi':
            if is_running_in_jupyter(): clear_output()
            print(self)

        

    def observation(self):
        """
        Provides an observation of the current state to the agent.
        This might be changed accordingly to the intended information
        the agent uses to decide.
        """
        from copy import deepcopy

        return deepcopy(self)


    def run_game(self,
                 policy : agents.Agent,
                 max_length   = None,
                 timeout_time = 0,
                 delay        = 0.,
                 seed         = None):
        """
        Run an entire game given a policy.

        # Arguments

        policy : Agent
            the policy being evaluated. Must inherit from Agent class
        
        max_length : int, default=None
            maximum number of steps before the game is stopped
        
        timeout : int, default=0
            computational limit, in seconds, for an agent to output an action.
            Letting it to be 0 means no timeout.
        
        delay : int, default=0
            time, in 
        """
        self.reset(seed=seed)

        experiences = []

        n_steps = 0
        obs = self.observation()
        while True:
            time.sleep(delay)

            with timeout(timeout_time):
                action = policy.act(obs)

            next_obs, reward, done = self.step(action)

            experiences.append((obs, action, reward, next_obs))

            n_steps += 1
            if max_length is not None and n_steps >= max_length: break
            if done: break

        return experiences
        
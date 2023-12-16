from pacman.distributions import UniformDistribution
from pacman.utils import discrete

class Actions:
    UP    = 'Up'
    DOWN  = 'Down'
    RIGHT = 'Right'
    LEFT  = 'Left'
    NOOP  = 'No action'

    actions = [UP, DOWN, RIGHT, LEFT]

    TURN_LEFT = {
        UP : LEFT,
        DOWN : RIGHT,
        RIGHT  : UP,
        LEFT  : DOWN,
        NOOP  : NOOP
    }

    TURN_RIGHT = dict([(y,x) for x, y in TURN_LEFT.items()])

    REVERSE = {
        UP : DOWN,
        DOWN : UP,
        RIGHT  : LEFT,
        LEFT  : RIGHT,
        NOOP  : NOOP
    }

    _directions = {
        UP : ( 0,  1),
        DOWN : ( 0, -1),
        RIGHT  : ( 1,  0),
        LEFT  : (-1,  0),
        NOOP  : ( 0,  0)
        }
    
    order = {
        UP : 0,
        DOWN : 1,
        RIGHT : 2,
        LEFT : 3
    }

    @staticmethod
    def reverse_direction(action):
        return Actions.REVERSE[action]

    @staticmethod
    def turn_right(action):
        return Actions.TURN_RIGHT[action]
    
    @staticmethod
    def turn_left(action):
        return Actions.TURN_LEFT[action]  

    @staticmethod
    def action_index(action):
        return Actions.order[action]

    @staticmethod
    def vector_to_action(vector):
        dx, dy = vector
        if dy > 0:
            return Actions.UP
        if dy < 0:
            return Actions.DOWN
        if dx < 0:
            return Actions.LEFT
        if dx > 0:
            return Actions.RIGHT
    
        return Actions.NOOP


    @staticmethod
    def action_to_vector(action, speed=1.):
        dx, dy =  Actions._directions[action]

        return (dx * speed, dy * speed)
    
    @staticmethod
    def apply_action(position, action):
        dx, dy = Actions.action_to_vector(action)

        x, y = position

        return (x + dx, y + dy)
    
    @staticmethod
    def calculate_next_position(position, action):
        x_int, y_int = discrete(position)
        dx, dy = Actions.action_to_vector(action)
        next_pos = discrete((x_int + dx, y_int + dy))

        return next_pos

    @staticmethod
    def sample() -> str:
        return UniformDistribution(Actions.actions).sample()
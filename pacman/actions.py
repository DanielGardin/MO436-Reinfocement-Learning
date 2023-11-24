from pacman.distributions import DiscreteDistribution

class Actions:
    UP    = 'Up'
    DOWN  = 'Down'
    RIGHT = 'Right'
    LEFT  = 'Left'
    NOOP  = 'No action'

    actions = [UP, DOWN, RIGHT, LEFT]

    TURN_LEFT = { UP : LEFT,
                  DOWN : RIGHT,
                  RIGHT  : UP,
                  LEFT  : DOWN,
                  NOOP  : NOOP}

    TURN_RIGHT = dict([(y,x) for x, y in TURN_LEFT.items()])

    REVERSE = { UP : DOWN,
                DOWN : UP,
                RIGHT  : LEFT,
                LEFT  : RIGHT,
                NOOP  : NOOP}

    _directions = {UP : ( 0,  1),
                   DOWN : ( 0, -1),
                   RIGHT  : ( 1,  0),
                   LEFT  : (-1,  0),
                   NOOP  : ( 0,  0)}

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
    def sample():
        return DiscreteDistribution.from_probs(Actions.actions, [1/len(Actions.actions) for _ in Actions.actions]).sample()
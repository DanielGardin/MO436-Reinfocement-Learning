from pacman.actions import Actions
from pacman.distributions import Distribution, DiscreteDistribution
from pacman.utils import discrete

class Agent:
    """
    A abstract class for an agent. The agent carries a policy, which is used
    to select an action given a current state. To implement a new agent, 
    create a subclass of this one and you must implement either `act` or 
    `get_distribution` methods.
    """

    def __init_subclass__(cls):
        def act(self, state):
            dist = self.get_distribution(state)

            return dist.sample()

    
        def get_distribution(self, state):
            selected_action = self.act(state)

            probs = [int(selected_action == action) for action in Actions.actions]

            return DiscreteDistribution.from_probs(Actions.actions, probs)


        has_act  = hasattr(cls, 'act')              or setattr(cls, 'act', act)
        has_dist = hasattr(cls, 'get_distribution') or setattr(cls, 'get_distribution', get_distribution)

        if not (has_act or has_dist):
            raise TypeError(f"Can't instantiate class '{cls.__name__}', " +
                        "without overriding at least on of the methods " +
                        "'act' or 'get_distribution'.")


class Ghost:
    GHOST_SPEED = 1.

    def __init__(self, initial_pos):
        self.initial_pos  = initial_pos
        self.position     = initial_pos
        self.scared_timer = 0
        self.direction    = Actions.NOOP

        self.name = self.__class__.__name__

    def __repr__(self):
        return f"{'scared ' if self.is_scared() else ''}{self.name} at {self.position}"

    def reset(self, pos=None):
        self.position     = self.initial_pos if pos is None else pos
        self.scared_timer = 0
        self.direction    = Actions.NOOP


    def apply_action(self, state, action):
        if action == Actions.NOOP:
            return

        legal_actions = state.get_legal_actions(self.position)

        if action not in legal_actions:
            raise Exception(f"Illegal ghost action {action}")

        speed = self.GHOST_SPEED
        if self.is_scared():
            self.scared_timer -= 1
            speed /= 2

        x, y = self.position
        dx, dy = Actions.action_to_vector(action, speed)

        self.position  = (x + dx, y + dy)
        self.direction = Actions.vector_to_action((dx, dy))


    def scare(self, scare_time):
        self.scared_timer = scare_time


    def is_scared(self):
        return self.scared_timer > 0


    def __init_subclass__(cls):
        def act(self, state):
            dist = self.get_distribution(state)

            return dist.sample()

    
        def get_distribution(self, state):
            selected_action = self.act(state)

            probs = [int(selected_action == action) for action in Actions.actions]

            return DiscreteDistribution.from_probs(Actions.actions, probs)


        has_act  = hasattr(cls, 'act')              or setattr(cls, 'act', act)
        has_dist = hasattr(cls, 'get_distribution') or setattr(cls, 'get_distribution', get_distribution)

        if not (has_act or has_dist):
            raise TypeError(f"Can't instantiate class '{cls.__name__}', " +
                        "without overriding at least on of the methods " +
                        "'act' or 'get_distribution'.")

class RandomGhost(Ghost):
    def get_distribution(self, state) -> Distribution:
        legal_actions = state.get_legal_actions(self.position)
        num_actions = len(legal_actions)

        reverse = Actions.reverse_direction(self.direction)

        if reverse in legal_actions and num_actions > 1:
            legal_actions.remove(reverse)

        return DiscreteDistribution({action : 1 for action in legal_actions})

class FollowGhost(Ghost):
    def act(self, state):
        x_int, y_int = discrete(self.position)
        lowest_dist = 1e10
        select_action = Actions.NOOP

        for action in state.get_legal_actions(self.position):
            dx, dy = Actions.action_to_vector(action)
            new_pos = discrete((x_int + dx, y_int + dy))

            distance = state.search_dist(new_pos, state.position)

            if distance < lowest_dist:
                lowest_dist = distance
                select_action = action
        
        return select_action


class TerritorialGhost(Ghost):
    pass



class ImmobileGhost(Ghost):
    def act(self, state):
        return Actions.NOOP
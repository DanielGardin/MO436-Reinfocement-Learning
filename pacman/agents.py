from pacman.actions import Actions
from pacman.distributions import Distribution, DiscreteDistribution, UniformDistribution
from pacman.utils import discrete
from typing import Generic, TypeVar
from random import random

class Agent:
    """
    A abstract class for an agent. The agent carries a policy, which is used
    to select an action given a current state. To implement a new agent, 
    create a subclass of this one and you must implement either `act` or 
    `get_distribution` methods.
    """

    def __init_subclass__(cls):
        def act(self, state) -> str:
            dist = self.get_distribution(state)

            return dist.sample()

    
        def get_distribution(self, state) -> Distribution[str]:
            selected_action = self.act(state)

            probs = [float(selected_action == action) for action in Actions.actions]

            return DiscreteDistribution(Actions.actions, probs)


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
        self.direction = action


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

            probs = [float(selected_action == action) for action in Actions.actions]

            return DiscreteDistribution(Actions.actions, probs)


        has_act  = hasattr(cls, 'act')              or setattr(cls, 'act', act)
        has_dist = hasattr(cls, 'get_distribution') or setattr(cls, 'get_distribution', get_distribution)

        if not (has_act or has_dist):
            raise TypeError(f"Can't instantiate class '{cls.__name__}', " +
                        "without overriding at least on of the methods " +
                        "'act' or 'get_distribution'.")

class FollowGhost(Ghost):
    """
    Deterministic ghost which selects the action that lead as close
    as possible to the agent.
    """
    def act(self, state):
        select_action = Actions.NOOP

        legal_actions = state.get_legal_actions(self.position)

        distances = {
            action : state.search_dist(Actions.calculate_next_position(self.position, action), state.position) for action in legal_actions
        }

        if self.is_scared():
            select_action = max(distances, key=lambda k : distances[k])
        
        else:
            select_action = min(distances, key=lambda k : distances[k])

        return select_action

class RandomGhost(Ghost):
    """
    Ghost that selects a random action every step, except turning 180°.
    """

    def get_distribution(self, state) -> Distribution:
        eps = random()

        if eps < 0.6:
            legal_actions = state.get_legal_actions(self.position)
            num_actions = len(legal_actions)

            reverse = Actions.reverse_direction(self.direction)

            if reverse in legal_actions and num_actions > 1:
                legal_actions.remove(reverse)

            return UniformDistribution(legal_actions)

        else:
            select_action = Actions.NOOP

            legal_actions = state.get_legal_actions(self.position)

            distances = {
                action : state.search_dist(Actions.calculate_next_position(self.position, action), state.position) for action in legal_actions
            }

            if self.is_scared():
                select_action = max(distances, key=lambda k : distances[k])
            
            else:
                select_action = min(distances, key=lambda k : distances[k])

            return UniformDistribution([select_action])


class ImmobileGhost(Ghost):
    """
    Ghost that makes no actions each step. Literally immobile
    """

    def act(self, state):
        return Actions.NOOP


class RobustGhost(Ghost):
    """
    Select the action which minimizes de distance from the agent supposing the agent
    takes the action to minimize distance from ghost
    """
    def act(self, state):
        robust_action = Actions.NOOP
        robust_distance = 0
        
        for agent_action in state.get_legal_actions(state.position):
            next_agent_pos = Actions.calculate_next_position(state.position, agent_action)

            distances = {
                action : state.search_dist(next_agent_pos, Actions.calculate_next_position(self.position, action)) \
                for action in state.get_legal_actions(self.position)
            }

            if self.is_scared():
                best_action = max(distances, key=lambda k : distances[k])
            
            else:
                best_action = min(distances, key=lambda k : distances[k])

            distance = distances[best_action]

            if distance >= robust_distance:
                robust_distance = distance
                robust_action   = best_action
        
        return robust_action

class StochasticRobustGhost(Ghost):
    """
    Select the action which minimizes de distance from the agent supposing the agent
    takes the action to minimize distance from ghost
    """
    def get_distribution(self, state):
        robust_actions = []
        robust_distance = 0
        
        for agent_action in state.get_legal_actions(state.position):
            next_agent_pos = Actions.calculate_next_position(state.position, agent_action)

            distances = {
                action : state.search_dist(next_agent_pos, Actions.calculate_next_position(self.position, action)) \
                for action in state.get_legal_actions(self.position)
            }

            if self.is_scared():
                best_action = max(distances, key=lambda k : distances[k])
            
            else:
                best_action = min(distances, key=lambda k : distances[k])

            distance    = distances[best_action]


            if distance >= robust_distance:
                robust_distance = distance
                robust_actions.append(best_action)
        
        return UniformDistribution(robust_actions)
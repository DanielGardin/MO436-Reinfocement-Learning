from collections import Counter
from random import random
from abc import ABC

class Distribution(ABC):
    """
    Base class of all probability distributions.
    """
    pass


class DiscreteDistribution(dict, Distribution):
    
    @classmethod
    def from_probs(cls, values, probs):
        assert sum(probs) == 1
        return cls(zip(values, probs))


    def mode(self):
        best_value = None
        best_prob  = 0.
        for value, prob in self.items():
            if prob > best_prob:
                best_prob  = prob
                best_value = value
        
        return best_value


    def sample(self):
        r = random()

        accum_prob = 0.

        for value, prob in self.items():
            accum_prob += prob
            if r <= accum_prob: return value
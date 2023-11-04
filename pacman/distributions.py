from collections import Counter
from random import random
from abc import ABC

class Distribution(ABC):
    """
    Base class of all probability distributions.
    """
    pass


class DiscreteDistribution(Counter, Distribution):
    
    @classmethod
    def from_probs(cls, values, probs):
        return cls(zip(values, probs))

    @property
    def probs(self):
        total = float(sum(self.values()))

        probs = dict()

        for key, value in self.items():
            probs[key] = value / total

        return probs

    def mode(self):
        return self.most_common(1)[0][0]
    
    def sample(self):
        r = random()

        accum_prob = 0.

        for value, prob in self.probs.items():
            accum_prob += prob
            if r <= accum_prob: return value
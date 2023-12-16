from collections import Counter
from numpy.random import rand
from abc import ABC, abstractmethod
from typing import TypeVar, List, Generic


T = TypeVar('T')
class Distribution(ABC, Generic[T]):
    """
    Base class of all probability distributions.
    """
    @abstractmethod
    def sample(self) -> T:
        pass
        


class DiscreteDistribution(Distribution[T]):
    def __init__(self, values:List[T], probs:List[float]) -> None:
        if sum(probs) != 1:
            probs = [prob/len(probs) for prob in probs]
        
        self.distribution = {value : prob for value, prob in zip(values, probs)}


    def mode(self) -> T:
        best_value = list(self.distribution.keys())[0]
        best_prob  = 0.
        for value, prob in self.distribution.items():
            if prob > best_prob:
                best_prob  = prob
                best_value = value
        
        return best_value


    def sample(self) -> T:
        r = rand()

        accum_prob = 0.

        for value, prob in self.distribution.items():
            accum_prob += prob
            if r <= accum_prob: return value
        
        return list(self.distribution.keys())[0]

class UniformDistribution(DiscreteDistribution):
    def __init__(self, values) -> None:
        uniform_probs = [1/len(values)] * len(values)

        super().__init__(values, uniform_probs)
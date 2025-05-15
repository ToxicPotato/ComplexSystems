from abc import ABC, abstractmethod

class BaseController(ABC):
    @abstractmethod
    def reset(self):
        """Reset any internal state between episodes."""
        pass

    @abstractmethod
    def select_action(self, obs):
        """Given an observation, return a discrete action."""
        pass

    def update(self, obs, action, reward, next_obs, done):
        """Optional online update (e.g. for learning agents)."""
        return
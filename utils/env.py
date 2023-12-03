import gym
import numpy as np
import random

from gym import spaces
from .bic import is_dag, calculate_bic_score
from collections import namedtuple, deque

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Define the Gym environment for the causal graph problem
class CausalGraphEnv(gym.Env):
    def __init__(self, n):
        super(CausalGraphEnv, self).__init__()
        self.n = n
        self.action_space = spaces.Discrete(n * n * 2)  # Two actions for each element in the matrix
        self.observation_space = spaces.Box(low=0, high=1, shape=(n, n), dtype=np.uint8)
        self.min_bic = 99999999
        self.best_state = np.zeros((self.n, self.n), dtype=np.uint8)

    def reset(self):
        self.state = np.zeros((self.n, self.n), dtype=np.uint8)
        return self.state

    def step(self, action, base_estimate):
        print("Action : ", action)
        i, j = divmod(action // 2, self.n)  # Determine the matrix element to change
        toggle = action % 2  # Determine whether to set or unset the edge

        # Save the current state to revert back if the action is invalid
        original_state = np.copy(self.state)
        # Apply the action
        if toggle == 1:
            self.state[i, j] = 1-self.state[i,j] 

        if is_dag(self.state):
            # Calculate the BIC score and determine the reward
            new_bic_score = calculate_bic_score(self.state, base_estimate)
            bic_score = - new_bic_score  # Reward is the negative BIC score
            done = False  # We can define the termination condition based on the problem
        else:
            # Revert to the original state as the action results in a non-DAG
            self.state = original_state
            bic_score = -1  # Penalize the action that results in a non-DAG
            done = True  # Optionally end the episode
        if bic_score < self.min_bic and bic_score > 0:
            self.best_state = np.copy(self.state)
            self.min_bic = bic_score
        return self.state, bic_score, done, {}

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(self.state)
    def result(self):
        return self.best_state

if __name__ == "__main__":
    memory = ReplayMemory(10)
    memory.push(1,2,3,4)
    memory.push(5,6,7,8)
    transitions = memory.sample(2)
    print(transitions[1].state)
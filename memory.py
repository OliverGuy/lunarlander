from collections import namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """Stores transitions to generate random batches from.

    Args:
        capacity (int): the maximum number of entries in the buffer.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition; overwrites oldest ones first if the buffer goes out of capacity.

        Args:
            state (batch(1) * channel * height * width tensor): the current state
            action (int): the action taken at the current state
            next_state (batch(1) * channel * height * width tensor): the state produced by (state, action)
            reward (float): the reward produced by (state, action)
        NB: done is characterised by next_step = None.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Returns a random sample from memory.

        Args:
            batch_size (int): Size of the requested sample.

        Returns:
            Transition list
        """
        return random.sample(self.memory, batch_size)

    def batch(self, batch_size):
        """Returns a randomised batch from memory; transposed version of sample.

        Args:
            batch_size (int): size of the requested batch

        Returns:
            Transition of batches: each dimension is a batch in itself
        """
        transitions = self.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        return Transition(*zip(*transitions))

    def __len__(self):
        return len(self.memory)

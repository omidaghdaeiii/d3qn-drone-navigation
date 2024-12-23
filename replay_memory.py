"""
Copyright (c) 2019 Phil Tabor
Copyright (c) 2024 Omid Aghdaei
Licensed under the MIT License. See the LICENSE file in the project root for more information.

This module implements a Replay Buffer for storing transitions in reinforcement learning.

The Replay Buffer stores experiences (state, action, reward, new state, done) and allows
for random sampling of batches of experiences for training purposes.

Classes:
    ReplayBuffer: Implements the functionality for storing and sampling transitions.
"""

import numpy as np

class ReplayBuffer:
    """
    A class to store and sample transitions for reinforcement learning.

    Attributes:
        mem_size (int): Maximum size of the memory.
        mem_cntr (int): Counter for the current number of stored transitions.
        state_memory (ndarray): Array to store the states.
        new_state_memory (ndarray): Array to store the new states.
        action_memory (ndarray): Array to store actions taken.
        reward_memory (ndarray): Array to store rewards received.
        terminal_memory (ndarray): Array to store terminal flags.
    """

    def __init__(self, max_size, input_shape):
        """
        Initialize the Replay Buffer.

        Args:
            max_size (int): Maximum number of transitions to store.
            input_shape (tuple): Shape of the input state.
        """
        self.mem_size = max_size
        self.mem_cntr = 0

        # Initialize memory for states, new states, actions, rewards, and terminal flags
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, new_state, done):
        """
        Store a transition in the Replay Buffer.

        Args:
            state (ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            new_state (ndarray): The new state after taking the action.
            done (bool): A flag indicating whether the episode has ended.
        """
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        """
        Sample a batch of transitions from the Replay Buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple: A tuple containing arrays of states, actions, rewards,
                   new states, and terminal flags for the sampled transitions.
        """
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminals

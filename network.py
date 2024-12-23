"""
Copyright (c) 2019 Phil Tabor
Copyright (c) 2024 Omid Aghdaei
Licensed under the MIT License. See the LICENSE file in the project root for more information.

This module implements a Dueling Deep Q-Network (DQN) using TensorFlow and Keras.

The Dueling DQN is a type of Q-network used in reinforcement learning. It consists of:
1. Convolutional layers for processing the input state.
2. Two separate fully connected streams: 
    - One for estimating the state-value function (V).
    - One for estimating the advantage function (A).

These two streams are combined to compute Q-values for each action, 
following the formula: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a'))).

Classes:
    DuelingDeepQNetwork: Implements the architecture and forward pass of the network.
"""

import tensorflow.keras as keras
from tensorflow.keras.layers import Conv2D, Dense, Flatten

class DuelingDeepQNetwork(keras.Model):
    """
    A Dueling Deep Q-Network (DQN) model that consists of convolutional layers followed by
    separate streams for value and advantage estimation.
    """

    def __init__(self, input_dims, n_actions):
        """
        Initialize the Dueling DQN.

        Args:
            input_dims (tuple): Dimensions of the input state.
            n_actions (int): Number of actions available in the environment.
        """
        super().__init__()
        self.conv_layer_1 = Conv2D(32, 8, strides=(4, 4), activation='relu',
                                    data_format='channels_first',
                                    input_shape=input_dims)
        self.conv_layer_2 = Conv2D(64, 4, strides=(2, 2), activation='relu',
                                    data_format='channels_first')
        self.conv_layer_3 = Conv2D(64, 3, strides=(1, 1), activation='relu',
                                    data_format='channels_first')
        self.flatten_layer = Flatten()
        self.fc_layer_1 = Dense(512, activation='relu')
        self.advantage_stream = Dense(n_actions, activation=None)
        self.value_stream = Dense(1, activation=None)

    def call(self, state):
        """
        Forward pass through the network.

        Args:
            state (tensor): The input state from the environment.

        Returns:
            tuple: The value and advantage estimates.
        """
        feature_map = self.conv_layer_1(state)
        feature_map = self.conv_layer_2(feature_map)
        feature_map = self.conv_layer_3(feature_map)
        feature_map = self.flatten_layer(feature_map)
        feature_map = self.fc_layer_1(feature_map)
        value_estimate = self.value_stream(feature_map)
        advantage_estimate = self.advantage_stream(feature_map)

        return value_estimate, advantage_estimate

    def summary(self):
        """
        Print a summary of the model's architecture.
        """
        return super().summary()

"""
Copyright (c) 2019 Phil Tabor
Copyright (c) 2024 Omid Aghdaei
Licensed under the MIT License. See the LICENSE file in the project root for more information.

Dueling Double Deep Q-Network (DDQN) Implementation

This module implements a Dueling Double Deep Q-Network (DDQN) agent for
reinforcement learning. The agent utilizes two neural networks:
one for evaluating the Q-values of actions and another (the target network)
for selecting the best action, following the principles of Double DQN.
Experience replay is employed to store past experiences, allowing the agent
to learn from a diverse set of state-action transitions.

Key Features:
- Dueling architecture to separate state value and advantage value.
- Experience replay buffer to enhance sample efficiency.
- Epsilon-greedy policy for action selection to balance exploration and exploitation.
- Target network to stabilize training and improve convergence.

Usage:
1. Initialize an Agent instance with the desired parameters.
2. Store transitions using `store_transition`.
3. Sample from memory and learn using the `learn` method in a training loop.
4. Save and load models using `save_models` and `load_models`.
"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import tensorflow.keras as keras
from network import DuelingDeepQNetwork
from replay_memory import ReplayBuffer

class Agent:
    """
    Represents an agent that utilizes a Dueling Double Deep Q-Learning algorithm 
    to learn from and interact with the environment. 

    Attributes:
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for the epsilon-greedy policy.
        lr (float): Learning rate for training the neural network.
        n_actions (int): Number of possible actions.
        input_dims (tuple): Dimensions of the input state.
        batch_size (int): Size of batches for sampling from memory.
        eps_min (float): Minimum epsilon for exploration.
        eps_dec (float): Rate at which epsilon decays.
        replace_target_cnt (int): Steps before updating the target network.
        algo (str): Name of the algorithm used.
        env_name (str): Name of the environment.
        chkpt_dir (str): Directory for saving model checkpoints.
        action_space (list): List of possible actions.
        learn_step_counter (int): Counter for learning steps.
        memory (ReplayBuffer): Experience replay buffer.
        fname (str): File name for saving models.
        q_eval (DuelingDeepQNetwork): Evaluation Q-network.
        q_next (DuelingDeepQNetwork): Target Q-network.
    """

    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None,
                 chkpt_dir='tmp/duelingddqn'):
        """
        Initializes the agent with the specified parameters for training.

        Args:
            gamma (float): Discount factor for future rewards.
            epsilon (float): Initial exploration rate for the epsilon-greedy policy.
            lr (float): Learning rate for the neural network.
            n_actions (int): Total number of actions available.
            input_dims (tuple): Dimensions of the input state (observations).
            mem_size (int): Size of the experience replay buffer.
            batch_size (int): Size of batches for training.
            eps_min (float): Minimum epsilon value for exploration (default=0.01).
            eps_dec (float): Decay rate for epsilon (default=5e-7).
            replace (int): Frequency of target network updates (in steps).
            algo (str): Optional; name of the algorithm used.
            env_name (str): Optional; name of the environment.
            chkpt_dir (str): Directory to save model checkpoints.
        """
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = list(range(n_actions))  # Updated line
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.fname = f"{self.chkpt_dir}{self.env_name}_{self.algo}_"

        # Evaluation and target networks
        self.q_eval = DuelingDeepQNetwork(input_dims, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr))
        self.q_next = DuelingDeepQNetwork(input_dims, n_actions)
        self.q_next.compile(optimizer=Adam(learning_rate=lr))

    def save_models(self):
        """Save the current weights of the evaluation and target networks."""
        self.q_eval.save(self.fname + 'q_eval')
        self.q_next.save(self.fname + 'q_next')
        print('... models saved successfully ...')

    def load_models(self):
        """Load the saved weights of the evaluation and target networks."""
        self.q_eval = keras.models.load_model(self.fname + 'q_eval')
        self.q_next = keras.models.load_model(self.fname + 'q_next')
        print('... models loaded successfully ...')

    def choose_action(self):
        """
        Choose an action based on the epsilon-greedy policy.

        Returns:
            int: Chosen action (0 in this placeholder implementation).
        """
        # Placeholder action selection logic
        action = 0
        return action

    def store_transition(self, state, action, reward, state_, done):
        """
        Store a transition in the replay buffer.

        Args:
            state (np.array): The initial state.
            action (int): Action taken in the initial state.
            reward (float): Reward received after the action.
            state_ (np.array): The next state after taking the action.
            done (bool): Whether the episode has ended.
        """
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        """
        Sample a batch of transitions from the replay buffer.

        Returns:
            tuple: A tuple of tensors (states, actions, rewards, new_states, dones).
        """
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )

        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        states_ = tf.convert_to_tensor(new_state)

        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        """Copy the weights from the evaluation network to the target network."""
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        """Decrement epsilon to reduce exploration over time."""
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min

    def learn(self):
        """
        Perform one step of training by sampling from the replay buffer and
        updating the evaluation network using gradient descent.
        """
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()

        # Sample a batch of transitions from memory
        states, actions, rewards, states_, dones = self.sample_memory()

        indices = tf.range(self.batch_size, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis=1)

        # Gradient descent step
        with tf.GradientTape() as tape:
            state_value, state_advantage = self.q_eval(states)
            next_state_value, next_state_advantage = self.q_next(states_)

            # Calculate the advantage function
            advantage = (
                state_value + state_advantage -
                tf.reduce_mean(state_advantage, axis=1, keepdims=True)
            )
            next_advantage = (
                next_state_value + next_state_advantage -
                tf.reduce_mean(next_state_value, axis=1, keepdims=True)
            )

            # Select the best action using the evaluation network (Double DQN)
            eval_advantage = advantage  # In Double DQN, action selection is from the evaluation network
            max_actions = tf.argmax(eval_advantage, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)

            q_next = tf.gather_nd(next_advantage, max_action_idx)
            q_pred = tf.gather_nd(advantage, action_indices)

            # Calculate the target Q-values
            q_target = (
                rewards + self.gamma * q_next * (1 - dones.numpy())
            )

            # Loss is the Mean Squared Error between predicted and target Q-values
            loss = keras.losses.MSE(q_pred, q_target)

        # Apply gradients to update the network
        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)
        self.q_eval.optimizer.apply_gradients(zip(grads, params))

        self.learn_step_counter += 1
        self.decrement_epsilon()

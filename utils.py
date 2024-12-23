"""
Copyright (c) 2019 Phil Tabor
Copyright (c) 2024 Omid Aghdaei
Licensed under the MIT License. See the LICENSE file in the project root for more information.

This module contains custom wrappers for a reinforcement learning environment using OpenAI Gym.
It includes functionality for managing memory, plotting learning curves, preprocessing frames,
stacking frames, and repeating actions within the environment.
"""

import os
from glob import glob

import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf

def manage_memory():
    """Configures TensorFlow to allow memory growth for GPUs to prevent memory allocation issues."""
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    """Plots the learning curve showing epsilon and scores over training steps.

    Args:
        x (list): The x-axis values representing training steps.
        scores (list): The scores achieved at each step.
        epsilons (list): The epsilon values used in the training.
        filename (str): The filename to save the plot.
        lines (list, optional): List of x-values to draw vertical lines. Defaults to None.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    num_scores = len(scores)
    running_avg = np.empty(num_scores)
    for t in range(num_scores):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

class RepeatActionAndMaxFrame(gym.Wrapper):
    """A Gym Wrapper that repeats actions and returns the maximum frame.

    This wrapper helps in environments where repeated actions are necessary
    to stabilize training, and it combines frames by taking the maximum value.

    Args:
        env (gym.Env): The environment to wrap.
        repeat (int): Number of times to repeat the action.
        clip_reward (bool): Flag to determine if the reward should be clipped.
        no_ops (int): Number of no-operation actions to perform before the actual action.
    """
    def __init__(self, env=None, repeat=0, clip_reward=False, no_ops=0):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 255, shape=(640, 480, 3), dtype=np.uint8)
        super().__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros_like((2, self.shape))
        self.clip_reward = clip_reward
        self.no_ops = no_ops

    def step(self, action):
        """Executes the given action and returns the maximum frame, total reward, and done status.

        Args:
            action: The action to execute.

        Returns:
            tuple: A tuple containing the maximum frame, total reward, done status, and additional
            info.
        """
        total_reward = 0.0
        done = False

        dir_path = r'D:\data\collision_dataset\validation'
        count_dirs = 0

        for (root, dirs, _) in os.walk(dir_path):
            count_dirs += len(dirs)  # Count directories

            for dirname in dirs:
                label_files = glob(f'{root}/{dirname}/*.txt')
                labels_list = []
                if label_files:
                    with open(label_files[0], 'r', encoding='utf-8') as txtfile:
                        lines = txtfile.readlines()
                        for line in lines:
                            labels_list.append(line.strip())

                images_list = glob(f'{root}/{dirname}/*.jpg')
                images_list.sort()

                for count_, (label, image_path) in enumerate(zip(labels_list, images_list)):
                    cv2.imread(image_path)  # Read the image
                    reward = label
                    if self.clip_reward:
                        reward = label
                    total_reward += reward
                    idx = count_ % 2
                    self.frame_buffer[idx] = reward

                done = True  # Set done to True after processing

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, total_reward, done, {}

class PreprocessFrame(gym.ObservationWrapper):
    """A Gym Wrapper that preprocesses frames for the environment.

    This class converts RGB frames to grayscale and resizes them for consistent input.

    Args:
        shape (tuple): The desired shape of the preprocessed frames.
        env (gym.Env): The environment to wrap.
    """
    def __init__(self, shape, env=None):
        super().__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=self.shape,
                                                dtype=np.float32)

    def observation(self, obs):
        """Processes the observation frame.

        Args:
            obs: The original observation frame.

        Returns:
            ndarray: The preprocessed observation frame.
        """
        gray_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_frame = cv2.resize(gray_frame, self.shape[1:], interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_frame, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0

        return new_obs

class StackFrames(gym.ObservationWrapper):
    """A Gym Wrapper that stacks frames for temporal observation.

    This class maintains a stack of previous observations to provide temporal context.

    Args:
        env (gym.Env): The environment to wrap.
        repeat (int): The number of frames to stack.
    """
    def __init__(self, env, repeat):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
                            env.observation_space.low.repeat(repeat, axis=0),
                            env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        """Resets the environment and the frame stack.

        Returns:
            ndarray: The stacked frames after reset.
        """
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, observation):
        """Updates the frame stack with the new observation.

        Args:
            observation: The current observation frame.

        Returns:
            ndarray: The updated stacked frames.
        """
        self.stack.append(observation)

        return np.array(self.stack).reshape(self.observation_space.low.shape)

def make_env(env_name, shape=(640, 480, 3), repeat=137, clip_rewards=False, no_ops=0):
    """Creates and returns a wrapped Gym environment.

    Args:
        env_name (str): The name of the environment to create.
        shape (tuple): The desired shape for preprocessing.
        repeat (int): Number of frames to repeat.
        clip_rewards (bool): Flag to clip rewards.
        no_ops (int): Number of no-operation actions.

    Returns:
        gym.Env: The wrapped environment.
    """
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, clip_rewards, no_ops)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)

    return env

"""
Copyright (c) 2019 Phil Tabor
Copyright (c) 2024 Omid Aghdaei
Licensed under the MIT License. See the LICENSE file in the project root for more information.

Main script to train a Dueling Double Deep Q-Learning (DQN) agent in a custom environment.

This script manages the agent's interaction with the environment, training over a 
specified number of episodes, and records performance metrics such as scores and 
epsilon values for further analysis. The script supports loading a pre-trained 
agent from a checkpoint and recording videos of the agent's performance.

Dependencies:
- numpy
- gym (with a specific environment)
- agent (custom implementation of the DQN agent)
- utils (utility functions for plotting and environment management)

Usage:
    Run this script directly. The agent will interact with the specified environment,
    training over a set number of games and optionally saving performance plots and 
    videos.
"""

import numpy as np
from gym import wrappers
from agent import Agent
from utils import plot_learning_curve, make_env, manage_memory

if __name__ == '__main__':
    # Manage memory for the agent
    manage_memory()

    # Create the environment
    env = make_env('DepEnv-v0')

    # Initialize constants
    BEST_SCORE = -np.inf
    LOAD_CHECKPOINT_FLAG = False
    RECORD_AGENT_FLAG = False
    N_GAMES_TO_TRAIN = 137

    # Initialize the agent
    agent = Agent(
        gamma=0.99,
        epsilon=1,
        lr=0.0001,
        input_dims=(env.observation_space.shape),
        n_actions=env.action_space.n,
        mem_size=50000,
        eps_min=0.1,
        batch_size=32,
        replace=1000,
        eps_dec=1e-5,
        chkpt_dir='models/',
        algo='DQNAgent',
        env_name='DepEnv-v0'
    )

    # Load model checkpoint if specified
    if LOAD_CHECKPOINT_FLAG:
        agent.load_models()
        agent.epsilon = agent.eps_min

    # Prepare filename for saving the plot
    PLOT_FILENAME = f"{agent.algo}_{agent.env_name}_lr{agent.lr}_{N_GAMES_TO_TRAIN}games"
    FIGURE_FILE_PATH = f"plots/{PLOT_FILENAME}.png"

    # Set up video recording if specified
    if RECORD_AGENT_FLAG:
        env = wrappers.Monitor(
            env, "video",
            video_callable=lambda episode_id: True,
            force=True
        )

    STEP_COUNT = 0  # Initialize step counter
    scores, eps_history, steps_array = [], [], []  # Lists to store metrics

    # Main training loop
    for episode in range(N_GAMES_TO_TRAIN):
        IS_EPISODE_DONE = False  # Reset done flag for each episode
        EPISODE_SCORE = 0  # Initialize score for this episode
        current_observation = env.reset()  # Reset environment

        while not IS_EPISODE_DONE:
            # Choose action based on the current observation
            ACTION_TAKEN = agent.choose_action(current_observation)
            # Take action and observe the result
            next_observation, reward, IS_EPISODE_DONE, info = env.step(ACTION_TAKEN)
            EPISODE_SCORE += reward  # Update score

            # Store transition and learn if not loading checkpoint
            if not LOAD_CHECKPOINT_FLAG:
                agent.store_transition(
                    current_observation,
                    ACTION_TAKEN,
                    reward,
                    next_observation,
                    IS_EPISODE_DONE
                )
                agent.learn()

            current_observation = next_observation  # Update observation
            STEP_COUNT += 1  # Increment step count

        scores.append(EPISODE_SCORE)  # Append score for the episode
        steps_array.append(STEP_COUNT)  # Append total steps

        # Calculate average score over the last 100 episodes
        avg_score = np.mean(scores[-100:])

        # Print episode information using f-string
        print(f'episode {episode} score {EPISODE_SCORE:.1f} avg score {avg_score:.1f} '
              f'best score {BEST_SCORE:.1f} epsilon {agent.epsilon:.2f} steps {STEP_COUNT}')

        # Save models if the score is the best so far
        if EPISODE_SCORE > BEST_SCORE:
            if not LOAD_CHECKPOINT_FLAG:
                agent.save_models()
            BEST_SCORE = EPISODE_SCORE  # Update best score

        eps_history.append(agent.epsilon)  # Append current epsilon value

    # Plot the learning curve
    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, FIGURE_FILE_PATH)

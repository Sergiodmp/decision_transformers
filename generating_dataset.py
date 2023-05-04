import torch
from torch.utils.data import Dataset
import gym
import numpy as np
import pandas as pd
import pickle

from prueba_env import env_va







def generate_data():




    env = env_va()
    obs = env.reset()

    obs_list = []
    action_list = []
    reward_list = []
    done_list = []
    info_list = []
    jobs_list = []
    current_time_step_list = []
    costcoils_list = []
    plants_df_list = []
    orders_df_list = []


        # Set the number of episodes and steps
    n_episodes = 10

    # Initialize the dataset
    dataset = []

    # Loop over the episodes
    for episode in range(n_episodes):

        # Reset the environment
        state = env.reset()

        for i in range(1000):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

            obs_list.append(obs)
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)
            jobs_list.append(env.jobs)
            current_time_step_list.append(env.current_time_step)
            costcoils_list.append(env.costcoils)
            plants_df_list.append(env.plants_df)
            orders_df_list.append(env.orders_df)

            if done:
                obs = env.reset()

        data = {"obs": obs_list,
                "action": action_list,
                "reward": reward_list,
                "done": done_list,
                "info": info_list,
                "jobs": jobs_list,
                "current_time_step": current_time_step_list,
                "costcoils": costcoils_list,
                "plants_df": plants_df_list,
                "orders_df": orders_df_list}
            # Save the dataset to a pickle file
        with open('env_va-dataset-v2.pkl', 'wb') as f:
            pickle.dump(data, f)
            
    # Convert the dataset to a pandas dataframe
    columns = ['state', 'action', 'next_state', 'reward']
    df = pd.DataFrame(dataset, columns=columns)



if __name__ == "__main__":
    generate_data()



'''import gym
import pickle
import csv
import logging
import numpy as np

'''
'''
from prueba_env import env_va
env = env_va()
dataset = []
env_name = 'env_va'

for episode in range(num_episodes):
    obs = env.reset()
    done = False
    while not done:
        action = agent.select_action(obs) # Replace with your own agent's action selection function
        next_obs, reward, done, info = env.step(action)
        dataset.append((obs, action, reward, next_obs))
        obs = next_obs


# Save the dataset to a pickle file
with open('env_va-dataset-v2.pkl', 'wb') as f:
    pickle.dump(dataset, f)
    '''

'''
def create_dataset(num_buffers, num_steps, game, data_dir_prefix, trajectories_per_buffer):
    # -- load data from memory (make more efficient)
    obss = []
    actions = []
    returns = [0]
    done_idxs = []
    stepwise_returns = []

    transitions_per_buffer = np.zeros(50, dtype=int)
    num_trajectories = 0
    while len(obss) < num_steps:
        buffer_num = np.random.choice(np.arange(50 - num_buffers, 50), 1)[0]
        i = transitions_per_buffer[buffer_num]
        print('loading from buffer %d which has %d already loaded' % (buffer_num, i))
        frb = FixedReplayBuffer(
            data_dir=data_dir_prefix + game + '/1/replay_logs',
            replay_suffix=buffer_num,
            observation_shape=(84, 84),
            stack_size=4,
            update_horizon=1,
            gamma=0.99,
            observation_dtype=np.uint8,
            batch_size=32,
            replay_capacity=100000)
        if frb._loaded_buffers:
            done = False
            curr_num_transitions = len(obss)
            trajectories_to_load = trajectories_per_buffer
            while not done:
                states, ac, ret, next_states, next_action, next_reward, terminal, indices = frb.sample_transition_batch(batch_size=1, indices=[i])
                states = states.transpose((0, 3, 1, 2))[0] # (1, 84, 84, 4) --> (4, 84, 84)
                obss += [states]
                actions += [ac[0]]
                stepwise_returns += [ret[0]]
                if terminal[0]:
                    done_idxs += [len(obss)]
                    returns += [0]
                    if trajectories_to_load == 0:
                        done = True
                    else:
                        trajectories_to_load -= 1
                returns[-1] += ret[0]
                i += 1
                if i >= 100000:
                    obss = obss[:curr_num_transitions]
                    actions = actions[:curr_num_transitions]
                    stepwise_returns = stepwise_returns[:curr_num_transitions]
                    returns[-1] = 0
                    i = transitions_per_buffer[buffer_num]
                    done = True
            num_trajectories += (trajectories_per_buffer - trajectories_to_load)
            transitions_per_buffer[buffer_num] = i
        print('this buffer has %d loaded transitions and there are now %d transitions total divided into %d trajectories' % (i, len(obss), num_trajectories))

    actions = np.array(actions)
    returns = np.array(returns)
    stepwise_returns = np.array(stepwise_returns)
    done_idxs = np.array(done_idxs)

    # -- create reward-to-go dataset
    start_index = 0
    rtg = np.zeros_like(stepwise_returns)
    for i in done_idxs:
        i = int(i)
        curr_traj_returns = stepwise_returns[start_index:i]
        for j in range(i-1, start_index-1, -1): # start from i-1
            rtg_j = curr_traj_returns[j-start_index:i-start_index]
            rtg[j] = sum(rtg_j)
        start_index = i
    print('max rtg is %d' % max(rtg))

    # -- create timestep dataset
    start_index = 0
    timesteps = np.zeros(len(actions)+1, dtype=int)
    for i in done_idxs:
        i = int(i)
        timesteps[start_index:i+1] = np.arange(i+1 - start_index)
        start_index = i+1
    print('max timestep is %d' % max(timesteps))

    return obss, actions, returns, done_idxs, rtg, timesteps
'''
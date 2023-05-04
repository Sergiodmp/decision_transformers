import datetime
import random
import torch

import pandas as pd
import gym
import numpy as np
import plotly.figure_factory as ff
import openpyxl
import operative_functions as asf
import globals

global results_2, winner_df, all_winner_df, snapshot

class env_va(gym.Env):
        def __init__(self):
                self.price_energy_consumption = 0.222 #euros/KWh
                self.index_va = ['va09', 'va10', 'va11', 'va11', 'va12', 'va12']
                self.df_parameters_energy=pd.DataFrame({
                                'a': [-4335, -4335, -8081.22, -141,-6011.6, -3855.45],
                                'b': [2.1, 2.1, 4.31, 3.27, 3.83, 2.4],
                                'c': [5405.53, 5405.53, 6826.2, 5943.73, 6742.25, 901.87],
                                'd': [191.27, 191.27, 240.12, 228.9, 195.85, 292.9],
                                'e': [212.31, 212.31, 319.5, 348.29, 264.99, 238.68],
                                'f': [9.44, 9.44, 12.68, 12.16, 11.74, 8.66]}, 
                                index=self.index_va)

                ############################################################################ instance variables of the class 
                ###############################basic variables

                self.snapshot = pd.read_excel(io='C:/Users/sergy/OneDrive/Documentos/MASTER/TFM/pruebas_sin_servidor/production_snapshot_pruebas.xlsx',
                                                        sheet_name='Hoja1', header=0, names = None, index_col = None,
                                                        usecols= 'A:T', engine= 'openpyxl')

                #print(self.snapshot)
                self.jobs=len(self.snapshot.axes[0]) #self jobs must reduce until 0
                print(self.jobs)

                '''We define the max time steps in the experiment file''' 

                #self.max_time_steps=1000 #maximum number of time steps in an episode. It is important because
                        #it limits the duration between the agent and the environment. Without this paprameter, the interaction
                        #could go on indefinitely, which would not be ideal for training the agent
                self.current_time_step=0
                self.action_space = gym.spaces.Discrete(self.jobs) #you can move from processing one coil to any other
                #print(self.action_space.n)
                self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.jobs,)) #the observation space is the current cost each time.
                        #This includes if each coil meets the plant rules, the difference of thickness...
                #####################################extra variables                 
                self.winner_df = pd.DataFrame()     
                self.all_winner_df =   pd.DataFrame()  
                self.assess_costs_coil = asf.va_bid_evaluation(self.df_parameters_energy,self.snapshot,self.price_energy_consumption,self.winner_df)
                self.jid_list_2=self.assess_costs_coil.loc[:,'Coil number example'].tolist()
                self.results_2 = asf.va_result(self.assess_costs_coil,self.jid_list_2)
                #print(results_2)
                #costcoils=np.random.rand(jobs)  #cost of the products randomly calculated and scalated between 0 and 1
                self.costcoils_df=self.results_2.loc[:,'cost']
                self.costcoils=self.costcoils_df.to_numpy()
                #print(costcoils)
                self.max_cost=self.results_2.iloc[-1]['cost']
                self.coil_with_min_cost=self.results_2.iloc[0]['Coil']


                self.plants_df=self.snapshot["PLANT"].value_counts()                    #dataframe that says the number of coils that each plant has to process
                self.plants_number=len(self.plants_df.axes[0])
                self.orders_df=self.snapshot["Production_Order_NR"].value_counts()      #dataframe that says the number of different orders of coils
                self.orders_number=len(self.orders_df.axes[0])
                act_dim=self.jobs
                #actions = torch.zeros((1, act_dim), device='cpu', dtype=torch.float32)
                #print(actions)
                #state = #reset_model()
                ###################################################actions are decided in the evaluate_episodes file
                '''                for t in range(self.max_time_steps):
                        

                        #############actions are made with torch
                        self.actions = torch.zeros((1, act_dim), device='cpu', dtype=torch.float32)
                        #actions = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)
                        self.action= torch.zeros_like(self.actions[-1])
                        self.actions[-1] = action
                        action = action.detach().cpu().numpy()
                        ##############tensors are a type of data, similar to numpy arrays but in this case, they can be run on gpu to accelerate the training.
                        self.state, self.reward, self.done, _ = self.step(self.action)
                        
                        
                        #print(winner_df)
                        '''

                '''
                observation_space = gym.spaces.Dict(
                        {
                        "action_mask": gym.spaces.Box(0, 1, shape=jobs + 1,)),
                        "real_obs": gym.spaces.Box(
                                low=0.0, high=1.0, shape=jobs, 22), dtype=float
                        ),
                        }
                )
                '''
                '''
                def _get_current_state_representation(self):
                state[:, 0] =legal_actions[:-1]
                        return {
                        "real_obs":state,
                        "action_mask":legal_actions,
                }

                def get_legal_actions(self):
                        returnlegal_actions
                '''
        def reset(self):
                self.current_time_step=0
                self.snapshot = pd.read_excel(io='C:/Users/sergy/OneDrive/Documentos/MASTER/TFM/pruebas_sin_servidor/production_snapshot_pruebas.xlsx',
                                                        sheet_name='Hoja1', header=0, names = None, index_col = None,
                                                        usecols= 'A:T', engine= 'openpyxl')

                self.winner_df = pd.DataFrame()     
                self.all_winner_df =   pd.DataFrame()  
                self.assess_costs_coil = asf.va_bid_evaluation(self.df_parameters_energy,self.snapshot,self.price_energy_consumption,self.winner_df)
                self.jid_list_2=self.assess_costs_coil.loc[:,'Coil number example'].tolist()
                self.results_2 = asf.va_result(self.assess_costs_coil,self.jid_list_2)
                #print(results_2)
                #costcoils=np.random.rand(jobs)  #cost of the products randomly calculated and scalated between 0 and 1
                self.costcoils_df=self.results_2.loc[:,'cost']
                self.costcoils=self.costcoils_df.to_numpy()
                #print(costcoils)
                self.max_cost=self.results_2.iloc[-1]['cost']
                self.coil_with_min_cost=self.results_2.iloc[0]['Coil']
                return self.costcoils


        def reward_scaler(self,reward):
                return reward /self.max_cost
                '''
                def reset_model(self):
                #########################basid variables
                costcoils=np.random.rand(jobs)
                current_time_step=0
                ##########################extra variables
                winner_df = pd.DataFrame()
                plants_df = pd.DataFrame()
                plants_number=0
                orders_df = pd.DataFrame()
                orders_num0ber=0
                costs_coil=pd.DataFrame()
                jid_list_2=list()
                results_2 = pd.DataFrame()
                winner_df = pd.DataFrame()
                '''
        def calculate_reward(self,action):
                reward=0
                reward= -1*self.costcoils[action]
                scaled_reward=self.reward_scaler(reward)

                return scaled_reward

        def step(self,action):
                #update the snapshot and winner_dataframe. We drop the coil with min cost and we added to the winner dataframe
                self.winner_df =self.results_2.loc[0,:]

                self.snapshot.drop(self.snapshot[(self.snapshot['Coil number example']==self.coil_with_min_cost)].index, inplace =True)
                self.snapshot = self.snapshot.reset_index(drop=True)
                print(self.winner_df)
                print("--------------------------------------------------------")
                self.all_winner_df = pd.concat([self.all_winner_df,self.winner_df])              
                self.all_winner_df =self.all_winner_df.reset_index(drop=True)
                #print(self.all_winner_df)
                #update current time step
                self.current_time_step+=1
                #calculate reward
                reward=self.calculate_reward(action)
                #update state
                self.assess_costs_coil = asf.va_bid_evaluation(self.df_parameters_energy,self.snapshot,self.price_energy_consumption,self.winner_df)
                self.jid_list_2=self.assess_costs_coil.loc[:,'Coil number example'].tolist()
                self.results_2 = asf.va_result(self.assess_costs_coil,self.jid_list_2)

                self.costcoils_df=self.results_2.loc[:,'cost']
                self.costcoils[action]=self.costcoils_df.to_numpy()[action]                    #cost of the action
                state=self.costcoils        
                #update_state(action)
                #check if the episode is done
                #done=self.current_time_step>self.max_time_steps
                if self.snapsht.empty:
                        done=True
                else:
                        done=False

                #create info dictionary
                info={}

                #return observation, reward,done and info
                return state, reward, done, info

                '''
                
                reward = 0.0
                if action ==jobs:
                        whilenon_assigned_jobs != 0:
                                reward -=increase_cost_step()
                        scaled_reward =_reward_scaler(reward)
                _check_no_op()
                        return (
                _get_current_state_representation(),
                        scaled_reward,
                _is_done(),
                        {},)
                '''








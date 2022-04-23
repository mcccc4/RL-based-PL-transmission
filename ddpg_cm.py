import numpy as np
from actor_net import ActorNet
from critic_net import CriticNet
from collections import deque
import random
from tensorflow_grad_inverter import grad_inverter

# Systen Parameters
UE_NUM = 10  # the number of users in the network
BS_NUM = 1  # the number of base stations. in the central of the window.
UE_AN = 1  # number of antennas per UE
BS_AN = 10  # number of antennas for BS, we should make it as 64 or 128 for Massive MIMO system
HIS_WIN = 10  # storage history widow for information at bs
delta = 0.01 # noise power
beta = 0.995  # reward penalty ,reward =  sumrate - penalty * prediction loss
PILOT_LEN = 13 # length of pilot signal

his_ce = HIS_WIN * UE_NUM * BS_AN # history information of channel estimation
his_fb = HIS_WIN * UE_NUM * BS_NUM # history information of user feedback (CSI at UE side)
his_re = HIS_WIN * UE_NUM * BS_NUM # history information of reward
his_recepilot = HIS_WIN * UE_NUM * BS_NUM # history information of received pilot
recepilot = UE_NUM * BS_NUM # received pilot at current TS
Pilot_signal = UE_NUM * PILOT_LEN # all the pilot information

epsilon_end_time = 100000

tPower_UE = -7 # 23 dBm = 0.1995 W = -7 dBW
tPower_BS = 5 # 35 dBm = 3.1623 W = 5 dBW
NOISE = -204 # -174 dBm/Hz = -204 dBW
Bandwidth = 100 * 10 ^6 # 100 MHz
PRB_band = 180000  # PRB-bandwidth = 180kHz
PRB_NUM = 556 # PRB_NUM * PRB_band = Bandwidth
Carrier_F = 6 * 10 ^ 9 # 6 GHz, carrier frequency in Hz
v = 30.0 # UT velocity in km/h

REPLAY_MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA=0.99

state_space = HIS_WIN * UE_NUM * (2 * BS_AN +2 ) + UE_NUM * PILOT_LEN
action_space = 2 * BS_AN * UE_NUM

is_grad_inverter = True
class DDPG:
    
    """ Deep Deterministic Policy Gradient Algorithm"""
    def __init__(self,ac_states, ac_actions):
        self.num_states = ac_states  #  HIS_WIN * UE_NUM * (2 * BS_AN +2 ) + UE_NUM * PILOT_LEN
        self.num_actions = ac_actions  #  2 * BS_AN * UE_NUM

        self.critic_net = CriticNet(self.num_states, self.num_actions)
        self.actor_net = ActorNet(self.num_states, self.num_actions)
        
        #Initialize Buffer Network:
        self.replay_memory = deque()
        
        #Intialize time step:
        self.time_step = 0
        self.counter = 0
        
        action_max = np.array(np.ones(action_space) * 2).tolist()
        action_min = np.array(np.ones(action_space) * (-2)).tolist()
        # action_nonzero = ACC_UE
        action_bounds = [action_max,action_min]
        self.grad_inv = grad_inverter(action_bounds)
        
        
    def evaluate_actor(self, state_t):
        return self.actor_net.evaluate_actor(state_t)

    def evaluate_predic(self, state_t):
        return self.predic_net.evaluate_predic(state_t)
    
    def add_experience(self, observation, nextobservation, action, reward, counter):
        self.observation = observation
        self.nextobservation = nextobservation
        self.action = action
        self.reward = reward
        self.counter = counter
        self.replay_memory.append((self.observation,self.nextobservation,self.action,self.reward,self.counter))
        self.time_step = self.time_step + 1
        if(len(self.replay_memory)>REPLAY_MEMORY_SIZE):
            self.replay_memory.popleft()
            
        
    def minibatches(self):
        batch = random.sample(self.replay_memory, BATCH_SIZE)
        #state t
        self.state_t_batch = [item[0] for item in batch]
        self.state_t_batch = np.array(self.state_t_batch)
        #state t+1
        self.predict_t_1_batch = [item[1] for item in batch]
        self.predict_t_1_batch = np.array(self.predict_t_1_batch)
        self.action_batch = [item[2] for item in batch]
        self.action_batch = np.array(self.action_batch)
        self.action_batch = np.reshape(self.action_batch,[len(self.action_batch),self.num_actions])
        self.reward_batch = [item[3] for item in batch]
        self.reward_batch = np.array(self.reward_batch)
        self.counter_batch = [item[4] for item in batch]
                  
                 
    def train(self):
        #sample a random minibatch of N transitions from R
        self.minibatches()
        self.action_t_1_batch = self.actor_net.evaluate_target_actor(self.predict_t_1_batch)
        #Q'(s_i+1,a_i+1)        
        q_t_1 = self.critic_net.evaluate_target_critic(self.predict_t_1_batch,self.action_t_1_batch)
        self.y_i_batch=[]
        terminal = False
        if self.time_step >= epsilon_end_time:
            terminal = True
        for i in range(0,BATCH_SIZE):
                           
            if terminal:
                self.y_i_batch.append(self.reward_batch[i])
            else:
                
                self.y_i_batch.append(self.reward_batch[i] + GAMMA*q_t_1[i][0])                 
        
        self.y_i_batch=np.array(self.y_i_batch)
        self.y_i_batch = np.reshape(self.y_i_batch,[len(self.y_i_batch),1])
        
        # Update critic by minimizing the loss
        self.critic_net.train_critic(self.state_t_batch, self.action_batch,self.y_i_batch)
        # Update actor proportional to the gradients:
        action_for_delQ = self.evaluate_actor(self.state_t_batch)  ##action for TS t

        print('size of actions', np.shape(action_for_delQ))
        if is_grad_inverter:
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch,action_for_delQ)#/BATCH_SIZE            
            self.del_Q_a = self.grad_inv.invert(self.del_Q_a,action_for_delQ) 
        else:
            self.del_Q_a = self.critic_net.compute_delQ_a(self.state_t_batch,action_for_delQ)[0]#/BATCH_SIZE
        
        # train actor network proportional to delQ/dela and del_Actor_model/del_actor_parameters:
        self.actor_net.train_actor(self.state_t_batch, self.del_Q_a)
 
        # Update target Critic and actor network
        self.critic_net.update_target_critic()
        self.actor_net.update_target_actor()

        #return loss_cost
        
        
        
                
        
        
        
                     
                 
        




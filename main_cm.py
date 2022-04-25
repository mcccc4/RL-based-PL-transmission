
"""
Created on Sep 25 12:30:21
"""

import numpy as np
import game_cm as game
from ddpg_cm import DDPG


# Systen Parameters
UE_NUM = 10  # the number of users in the network
BS_NUM = 1  # the number of base stations. in the central of the window.
UE_AN = 1  # number of antennas per UE
BS_AN = 10  # number of antennas for BS, we should make it as 64 or 128 for Massive MIMO system
HIS_WIN = 5  # storage history widow for information at bs
delta = 0.01 # noise power
beta = 0.995  # reward penalty ,reward =  sumrate - penalty * prediction loss
PILOT_LEN = 13 # length of pilot signal

his_ce = HIS_WIN * UE_NUM * BS_AN # history information of channel estimation
his_fb = HIS_WIN * UE_NUM * BS_NUM # history information of user feedback (CSI at UE side)
his_re = HIS_WIN * UE_NUM * BS_NUM
his_recepilot = HIS_WIN * UE_NUM * BS_NUM # history information of received pilot
recepilot = UE_NUM * BS_NUM # received pilot at current TS
Pilot_signal = UE_NUM * PILOT_LEN

epsilon_end_time = 50000

tPower_UE = -7
tPower_BS = 5
NOISE = -204
Bandwidth = 100 * 10 ^6
PRB_band = 180  # PRB-bandwidth = 180kHz
PRB_NUM = 556
Carrier_F = 6 * 10 ^ 9

# convert all the matrix states into arrays for NN inputs
def preprocess(bs_list):  ### before input to the DNN, flatten all the information
    his_ce = bs_list[0]
    his_fb = bs_list[1]
    his_re = bs_list[2]
    his_recepilot = bs_list[3]
    pilot_signal = bs_list[5]

    hisce_in = his_ce.flatten()
    hisfb_in = his_fb.flatten()
    hisre_in = his_re.flatten()
    his_recepilot_in = his_recepilot.flatten()
    pilot_signal_in = prepro_complex(pilot_signal)
    hisfb_in = prepro_complex(hisfb_in)

    array1 = np.hstack((hisce_in, hisfb_in))
    array2 = np.hstack((array1, hisre_in))
    array3 = np.hstack((array2, his_recepilot_in))
    his_record = np.hstack((array3, pilot_signal_in))

    return his_record

def prepro_complex(input_matrix):

    matrix_in = input_matrix.flatten()  # flatten
    length = np.size(matrix_in)
    matrix_real = np.zeros((length), dtype=float)
    matrix_imag= np.zeros((length), dtype=float)
    for i in range(np.size(matrix_in)):
        data = complex(matrix_in[i])
        matrix_real[i] = np.real(data)
        matrix_imag[i] = np.imag(data)

    matrix_input = np.hstack((matrix_real, matrix_imag))

    return matrix_input


def main():

    # init Game
    EHstates = game.GameState()
    user_list, bs_list = EHstates.init_system()
    his_record = preprocess(bs_list)
    # ddpg input
    state_space = HIS_WIN * UE_NUM * (5 * BS_AN + 1 ) + 2 * UE_NUM * PILOT_LEN
    action_space = 2 * BS_AN * UE_NUM
    agent = DDPG(state_space, action_space) # (state_space, action_space)
    num_states = state_space
    counter=0
    # saving reward:
    total_reward_his = np.zeros((epsilon_end_time))

    # run the game

    for i in range(epsilon_end_time):
        print("==== Starting episode no:",i,"====","\n")
        Observation = his_record
        print('size of observation', np.size(Observation))
        action = agent.evaluate_actor(np.reshape(Observation, [1, num_states]))
        rate_reward, prediction_loss, next_bs_list, Total_reward = EHstates.frame_update(action, i)

        his_record = preprocess(next_bs_list)
        nextObservation = np.hstack((his_record))

        # add s_t,s_t+1,action,reward to experience memory
        agent.add_experience(Observation, nextObservation, action, Total_reward, counter)

        # train critic and actor network
        if counter > 64:
            agent.train()
        #save
        total_reward_his[counter] = Total_reward
        counter += 1

        # check if episode ends:
        if (counter == epsilon_end_time):
            print('EPISODE: ', counter )
            print("Printing reward to file")
            np.savetxt('total_reward_his.txt', total_reward_his, delimiter=',')
            print('\n\n')

if __name__ == '__main__':
    main()    
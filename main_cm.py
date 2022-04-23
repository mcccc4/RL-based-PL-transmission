#Implementation of Deep Deterministic Gradient with Tensor Flow"
"""
Created on Sep 25 12:30:21 2020

@author: chum
"""

import numpy as np
import game_cm as game
from ddpg_cm import DDPG
# from LSTMDQN_cm import LSTMDQN

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
his_re = HIS_WIN * UE_NUM * BS_NUM # history information of reward
his_recepilot = HIS_WIN * UE_NUM * BS_NUM # history information of received pilot
recepilot = UE_NUM * BS_NUM # received pilot at current TS
Pilot_signal = UE_NUM * PILOT_LEN # all the pilot information

epsilon_end_time = 100000

tPower_UE = -7 # 23 dBm = 0.1995 W = -7 dBW
tPower_BS = 5 # 35 dBm = 3.1623 W = 5 dBW
NOISE = -204 # -174 dBm/Hz = -204 dBW
Bandwidth = 100 * 10 ^6 # 100 MHz
PRB_band = 180  # PRB-bandwidth = 180kHz
PRB_NUM = 556 # PRB_NUM * PRB_band = Bandwidth
Carrier_F = 6 * 10 ^ 9 # 6 GHz, carrier frequency in Hz
v = 30.0 # UT velocity in km/h


# convert all the matrix states into arrays for NN inputs
def preprocess(bs_list):  ### before input to the DNN, flatten all the information
    # inputs for prediction CSI, history information, bs_list[0] - bs_list[5]
    his_ce = bs_list[0]  # prediction history
    his_fb = bs_list[1]  #
    his_re = bs_list[2]  #
    his_recepilot = bs_list[3]
    pilot_signal = bs_list[5]

    hisce_in = his_ce.flatten() # 1 * ( HIS_WIN * 2 * UE_NUM * BS_AN)   +
    hisfb_in = his_fb.flatten() # 1 * (HIS_WIN * UE_NUM * BS_AN)
    hisre_in = his_re.flatten() # 1 * (HIS_WIN * UE_NUM)     +
    his_recepilot_in = his_recepilot.flatten() # 1 * ( HIS_WIN * UE_NUM * BS_AN)    +
    pilot_signal_in = prepro_complex(pilot_signal)   # UE_NUM * PILOT_LEN *2      +
    hisfb_in = prepro_complex(hisfb_in)  # 1 * (HIS_WIN * UE_NUM * BS_AN) *2      +

    array1 = np.hstack((hisce_in, hisfb_in))
    array2 = np.hstack((array1, hisre_in))
    array3 = np.hstack((array2, his_recepilot_in))
    his_record = np.hstack((array3, pilot_signal_in))

    return his_record

def prepro_complex(input_matrix):

    matrix_in = input_matrix.flatten()  # flattenc
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

    # init EH Game
    EHstates = game.GameState()
    user_list, bs_list = EHstates.init_system()
    his_record = preprocess(bs_list)

    # ddpg input is num_state = predicted battery states, num_actions = ue_num
    state_space = HIS_WIN * UE_NUM * (5 * BS_AN + 1 ) + 2 * UE_NUM * PILOT_LEN
    action_space = 2 * BS_AN * UE_NUM ## 2 means the real part and the image part are seperated
    agent = DDPG(state_space, action_space) # (state_space, action_space)
    num_states = state_space
    counter=0
    # saving reward:
    total_reward_his = np.zeros((epsilon_end_time))
    predicloss_his = np.zeros((epsilon_end_time))
    rate_his = np.zeros((epsilon_end_time))
    trainloss_his = np.zeros((epsilon_end_time))

    # run the game

    for i in range(epsilon_end_time):
        print("==== Starting episode no:",i,"====","\n")
        Observation = his_record
        print('size of observation', np.size(Observation))
        action = agent.evaluate_actor(np.reshape(Observation, [1, num_states]))
        #print("Action size at TS", counter, " :", action, "\n")

        rate_reward, prediction_loss, next_bs_list, Total_reward = EHstates.frame_update(action, i)

        his_record = preprocess(next_bs_list)
        nextObservation = np.hstack((his_record))

        # add s_t,s_t+1,action,reward to experience memory
        agent.add_experience(Observation, nextObservation, action, Total_reward, counter)

        # trueChannel = np.ndarray(next_bs_list[1][0])
        # loss_current = brain.setPerception(nextObservation, treward, trueChannel)

        # train critic and actor network
        if counter > 64:
            agent.train()
            # print('training starts', str(counter))
            # train_loss = agent.train()
            # trainloss_his[counter] = train_loss
        #save
        total_reward_his[counter] = Total_reward
        predicloss_his[counter] = prediction_loss
        rate_his[counter] = rate_reward
        counter += 1

        # check if episode ends:
        if (counter == epsilon_end_time):
            print('EPISODE: ', counter )
            print("Printing reward to file")
            np.savetxt('total_reward_his.txt', total_reward_his, delimiter=',')
            np.savetxt('predic_loss_his.txt', predicloss_his, delimiter=',')
            np.savetxt('rate_his.txt', rate_his, delimiter=',')
            #np.savetxt('trainloss_his.txt', trainloss_his, delimiter=',')
            print('\n\n')

if __name__ == '__main__':
    main()    
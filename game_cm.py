
"""
Created on Sep 25 12:30:21
Author: C
"""

import random
import numpy as np
import sys

UE_NUM = 10  # the number of users in the network
BS_NUM = 1  # the number of base stations. in the central of the window.
UE_AN = 1  # number of antennas per UE
BS_AN = 10  # number of antennas for BS, we should make it as 64 or 128 for Massive MIMO system
HIS_WIN = 5  # storage history widow for information at bs
delta = 0.01 # noise power
beta = 0.995
PILOT_LEN = 13

his_ce = HIS_WIN * UE_NUM * BS_AN # history information of channel estimation
his_fb = HIS_WIN * UE_NUM * BS_NUM # history information of user feedback (CSI at UE side)
his_re = HIS_WIN * UE_NUM * BS_NUM # history information of reward

epsilon_end_time = 50000


tPower_UE = -7
tPower_BS = 5
NOISE = -204
Bandwidth = 100 * 10 ^6
PRB_band = 180  # PRB-bandwidth = 180kHz
PRB_NUM = 556
Carrier_F = 6 * 10 ^ 9


class GameState:

  def __init__(self):
      self.user_list = init_users()
      self.bs_list = init_bs()
      self.channel = load_channel()
      self.Sumrate = np.zeros(1 * UE_NUM)
      self.Preloss = np.zeros(1 * UE_NUM)

  def init_system(self):
      user_init = init_users()
      bs_init = init_bs()
      self.channel = load_channel()
      return user_init, bs_init


  def frame_update(self, action, timeslot):
      """
      after uplink trans and downlink trans, getting the reward, then frame update
      use the action(preCSI) and feedback information to generate the new environment states (new frame) for agent BS
      """
      self.bs_list = gen_uplink(self, timeslot)
      self.Sumrate, self.PreLoss, Total_reward = cal_reward(self, action, timeslot)
      self.user_list = user_update(self)
      self.bs_list = his_store2(self, action, timeslot)

      rate_reward = np.sum(self.Sumrate)
      prediction_loss = np.sum(self.PreLoss)
      bs_list = self.bs_list

      return rate_reward, prediction_loss, bs_list, Total_reward
"""
functions that are used in the above class 
"""

def gen_uplink(self, timeslot):

    bs_list = bs_update(self, timeslot)
    bs_list = his_store1(bs_list)

    return bs_list

def get_curr_chan(self, ts_num):
    all_channel = self.channel

    channel = all_channel
    current_channel = list()
    for user in range(UE_NUM):
        chan_ue = channel[user]
        curr_chan = chan_ue[(ts_num * 10): ((ts_num + 1) * 10)]
        curr_chan = np.asarray(curr_chan)
        current_channel.append(curr_chan)

    return current_channel


def user_update(self):
    """
    UEs receive signals from BS every transmission TS,  user_list[5],
    """
    user_list = self.user_list
    sum_rate = self.Sumrate
    for user in range(UE_NUM):
        user_list[user][5] = sum_rate[user]
    return user_list


def bs_update(self, ts_num):
    """
    Uplink transmission finishes, BS receives the pilot signals
    Then, update new received pilot signals, bs_list
    get ready for DNN input, history information does not need to change
    """
    bs_list = self.bs_list
    ts = ts_num
    bs_list[4] = uplink_trans(self, ts)
    return bs_list


def uplink_trans(self, ts_num):
    """
     The uplink transmission process, including 13 pilot symbols and 1 data symbols.
     Assume the channel does not change during one learning TS (that is, 14 transmission TS)
    """
    ts = ts_num
    downlink = get_curr_chan(self, ts)  # size is UE_NUM * (channel)
    user_list = self.user_list
    received_p = np.zeros((BS_AN, UE_NUM), dtype=float)  # received pilot signals BS_AN * UE_NUM
    user_data = gen_ue_signal()
    for ue in range(UE_NUM):
        user_list[ue][4] = user_data[ue]
        cha_downlink = downlink[ue]
        pilot_sig = user_list[ue][3]
        for antenna in range(BS_AN):
            ch_down = cha_downlink[antenna]
            receive = 0.0
            for i in range(PILOT_LEN):
                receive += ch_down * pilot_sig[i]

            receive += ch_down * user_list[ue][4]

            received_p[antenna][ue] = abs(receive) * tPower_BS - NOISE

    return received_p


def his_store1(bs_list):
    ## when the uplink finishes, update partial history information.
    """
     The uplink transmission process, including 13 pilot symbols and 1 data symbols.
     Assume the channel does not change during one learning TS (that is, 14 transmission TS)
    """
    bs_list = bs_list

    # move one by one, leave room for new values at position [0]
    for count in range(-HIS_WIN + 1, -1):
        bs_list[3][-count] = bs_list[3][-count - 1]  # received pilot history

    bs_list[3][0] = bs_list[4]

    return bs_list


def his_store2(self, action, timeslot):  ##  update all the history information.
    """
     The uplink transmission process, including 13 pilot symbols and 1 data symbols.
     Assume the channel does not change during one learning TS (that is, 14 transmission TS)
    """
    bs_list = self.bs_list
    user_list = self.user_list

    current_cha = action
    # move one by one, leave room for new values at position [0]
    for count in range(-HIS_WIN + 1, -1):
        bs_list[0][-count] = bs_list[0][-count - 1]  # CP history
        bs_list[1][-count] = bs_list[1][-count - 1]  # feedback  history
        bs_list[2][-count] = bs_list[2][-count - 1]  # reward history

    bs_list[0][0] = current_cha

    real_channel = get_curr_chan(self, timeslot)
    for ue in range(UE_NUM):
        user_csi = real_channel[ue]
        bs_list[1][0][ue] = user_csi  # store feedback csi
        # store reward
        user_rate = user_list[ue][5]
        bs_list[2][0][ue] = user_rate

    return bs_list


def cal_reward(self, action, ts):

    real_channel = get_curr_chan(self, ts)
    ## sum rate
    Sum_rate = np.zeros((UE_NUM, BS_NUM), dtype=float)  ## sum rate from every user
    Loss = np.zeros((UE_NUM, BS_NUM), dtype=float)  ## sum rate from every user
    BS_tpower = tPower_BS
    down_data = gen_bs_signal()  # list， size UE_NUM * 1
    trans_seq = np.transpose(down_data)  # 1 * UE_NUM (10)
    power_symbol = np.sqrt(trans_seq)

    es_channel = np.zeros((UE_NUM, BS_AN), dtype=complex)
    for ue in range(UE_NUM):
        ## getting the action channel
        start = ue * 10
        end = (ue+1)*10
        action_useful = action[0]
        real = action_useful[start:end]
        image = action_useful[(start+100):(end+100)]
        channel_e = real + 1j*image
        es_channel[ue] = channel_e
    # Getting beamforming matrix
    beam_matrix = beam_forming(es_channel, power_symbol)
    # computing the rate
    for user in range(UE_NUM):
        eff_data = trans_seq[user] * beam_matrix[user]
        ## data transmission
        received_sum = 0
        for antenna in range(BS_AN):
            received_sum += eff_data[antenna] * real_channel[user][antenna]
        ## SNR
        snr = BS_tpower * abs(received_sum) - NOISE
        rate = PRB_band * np.log2(1 + 10 ** (snr / 10))  # Shannon rate formula
        Sum_rate[user] = rate
        ## prediction loss
        differ = np.subtract(es_channel[user], real_channel[user])
        differ = np.sum(differ)
        Loss[user] = abs(differ)

    total_reward = np.sum(Sum_rate)


    return Sum_rate, Loss, total_reward

def load_pilot():
    pilot = list()
    for i in range(UE_NUM):
        ## store PILOT_LEN sequence for user i
        data = r'C:\Users...'.format(number=str(i + 1))
        f = open(data)
        per_user_pilot = list()
        for line in f:
            line1 = line.replace('i', 'j').split()
            line2 = line1[0] # string
            line2 = line2.replace(" ", "")
            line3 = line2.replace("+-", "-")
            data_line = complex(line3)
            per_user_pilot.append(data_line)

        pilot.append(per_user_pilot)

    return pilot

def load_channel():
    channel_D = list()  # two dimensions
    for i in range(UE_NUM):  # UE i
        id = r'C:\Users...'.format(number=str(i + 1))
        f = open(id)
        per_user_channel = list()
        for line in f:
            line1 = line.replace('i', 'j').split()
            for data in line1:
                data = data.replace(" ", "")
                data1 = data.replace("+-", "-")
                data_value = complex(data1)
                per_user_channel.append(data_value)

        channel_D.append(per_user_channel)

    return channel_D

def gen_ue_signal():
    ## generate uplink transmission
    uplink_s = list()
    for i in range(UE_NUM):
        real = np.random.normal(loc=0.0, scale=1.0, size=1)
        image = np.random.normal(loc=0.0, scale=1.0, size=1)
        signal = complex(real, image)
        uplink_s.append(signal)

    return uplink_s

def gen_bs_signal():
    ## generate downlink transmission
    downlink_s = list()
    for i in range(UE_NUM):
        real = np.random.normal(loc=0.0, scale=1.0, size=1)
        image = np.random.normal(loc=0.0, scale=1.0, size=1)
        signal = complex(real, image)
        downlink_s.append(signal)
    return downlink_s

def init_users():
    """
    initialize user lists:
    0) user transmission power
    1) # of antennas
    2) PILOT_LEN
    3) user pilot signals
    4) initial transmission data = 1
    5) received signal
    return: user_list: (0) - (6)
    """
    ## load all the pilots
    pilot_sig = load_pilot()
    ## put the signals to each UE list
    user_list = list() # two dimensions
    for i in range(UE_NUM):  # user
        user = list()  # for user i
        user.append(tPower_UE)  # transmission power of UE i at position 0
        user.append(1) # single antenna per UE at position 1
        user.append(PILOT_LEN) # PILOT length of UE at position 2
        temp = pilot_sig[i]
        temp = np.asarray(temp)  ## change a list into an array
        user.append(temp) # UE i's pilot sequence at position 3
        user.append(1.0) # initialized transmission data at position 4 (change)
        user.append(0.0) # initialized received data (reward) at position 5 (change)
        user_list.append(user)
    return user_list

def init_bs():
    """
    base station information contains:
    (0) channel prediction history #
    (1) feedback channel history  #
    (2) reward history  #
    (3) received pilot history #
    (4) current received pilot  #
    (5) Pilot signals
    (6) transmission power
    (7) # of antennas
    """
    tPower = tPower_BS
    his_ce = np.zeros((HIS_WIN, 2*UE_NUM*BS_AN), dtype=float)
    his_fb = np.zeros((HIS_WIN, UE_NUM, BS_AN), dtype=complex)
    his_re = np.zeros((HIS_WIN, UE_NUM), dtype=float)
    his_recepilot = np.zeros((HIS_WIN, BS_AN, UE_NUM), dtype=float)
    recepilot = np.zeros((BS_AN, UE_NUM), dtype=float)

    pilot_sig = load_pilot() # list
    pilot_signals = list()
    for user in range(UE_NUM):
        for i in range(PILOT_LEN):
            pilot_signals.append(pilot_sig[user][i])
    Pilot_signal = np.asarray(pilot_signals)
    ## add information to the bs list
    bs_list = list()
    bs_list.append(his_ce) # 0
    bs_list.append(his_fb) # 1
    bs_list.append(his_re) # 2
    bs_list.append(his_recepilot) # 3
    bs_list.append(recepilot) # 4
    bs_list.append(Pilot_signal)  # 5 array
    bs_list.append(tPower) # 6
    bs_list.append(BS_AN) # 7

    return bs_list

def beam_forming(channel, power):
    # channel  with size BS_AN * 1,
    # ZF for channel beam-forming
    power_s = power
    power_symbol = (np.sum(power_s**2))/UE_NUM
    power_symbol = power_symbol/2
    down_ch = channel
    Beita = np.sqrt( (power_symbol * BS_AN) / np.trace( np.linalg.inv( np.dot(down_ch, np.transpose(down_ch)) ) ) * power_symbol)
    beam_matrix = Beita * np.transpose(down_ch) * np.linalg.inv( np.dot(down_ch, np.transpose(down_ch)) )

    return beam_matrix # contain all the UEs' preodin

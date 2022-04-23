#Reference:
#https://github.com/MOCR/

import tensorflow as tf
import numpy as np
BATCH_SIZE = 64
UE_NUM = 30


class grad_inverter:
    def __init__(self, action_bounds, nonzero_num):

        self.sess = tf.InteractiveSession()       
        
        self.action_size = len(action_bounds[0])
        
        self.action_input = tf.placeholder(tf.float32, [None, self.action_size]) ## input the action
        self.pmax = tf.constant(action_bounds[0], dtype = tf.float32)
        self.pmin = tf.constant(action_bounds[1], dtype = tf.float32)
        self.nonzero_num = nonzero_num
        self.prange = tf.constant([x - y for x, y in zip(action_bounds[0],action_bounds[1])], dtype = tf.float32)
        self.pdiff_max = tf.div(-self.action_input+self.pmax, self.prange)
        self.pdiff_min = tf.div(self.action_input - self.pmin, self.prange)
        self.zeros_act_grad_filter = tf.zeros([self.action_size]) ## all zeros
        self.act_grad = tf.placeholder(tf.float32, [None, self.action_size]) ## input the gradient
        self.grad_inverter = tf.where(tf.greater(self.act_grad, self.zeros_act_grad_filter), tf.multiply(self.act_grad, self.pdiff_max), tf.multiply(self.act_grad, self.pdiff_min))
    
    def invert(self, grad, action):
        self.invert_grad = self.sess.run(self.grad_inverter, feed_dict = {self.action_input: action, self.act_grad: grad[0]})
        print('size of grad', np.shape(grad[0]))
        zero_num = tf.count_nonzero(self.invert_grad)
        self.final_grad = self.invert_grad
        print('size of final_grad0', np.shape(self.final_grad))
        if zero_num == UE_NUM - self.nonzero_num:
            self.final_grad = tf.zeros(BATCH_SIZE, UE_NUM)
        print('size of final_grad', np.shape(self.final_grad))
        return self.final_grad



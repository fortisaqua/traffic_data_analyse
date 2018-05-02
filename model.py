import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np

class Model():
    def __init__(self, args, training=True):
        self.args = args
        self.training = training

    def build_model(self):
        if self.args.model == 'lstm':
            self.create_lstm()

    def create_lstm(self):
        self.X = tf.placeholder(
            tf.float32, [self.args.batch_size,self.args.time_step,self.args.input_size])
        self.Y = tf.placeholder(
            tf.float32, [self.args.batch_size, self.args.time_step, self.args.output_size])

        self.pred,self.final_state = self.lstm(self.X)
        self.loss = self.SmoothL1(self.pred,self.Y,1)

    def lstm(self,X):
        # 输入层、输出层权重、偏置
        weights = {
            'in': tf.Variable(tf.random_normal([self.args.input_size, self.args.rnn_size])),
            'out': tf.Variable(tf.random_normal([self.args.rnn_size, 1]))
        }
        biases = {
            'in': tf.Variable(tf.constant(0.1, shape=[self.args.rnn_size, ])),
            'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
        }

        batch_size = tf.shape(X)[0]
        time_step = tf.shape(X)[1]
        w_in = weights['in']
        b_in = biases['in']
        input = tf.reshape(X,[-1,self.args.input_size])
        input_rnn = tf.matmul(input,w_in)+b_in
        input_rnn = tf.reshape(input_rnn,[batch_size, time_step, self.args.rnn_size])
        cells = []
        for _ in range(self.args.num_layers):
            with tf.variable_scope("lstm_cell_"+str(_)):
                sub_cell = rnn.BasicLSTMCell(self.args.rnn_size)
                if self.training and (self.args.output_keep_prob < 1.0 or self.args.input_keep_prob < 1.0):
                    sub_cell = rnn.DropoutWrapper(sub_cell,input_keep_prob=self.args.input_keep_prob,
                                              output_keep_prob=self.args.output_keep_prob)
                cells.append(sub_cell)
        self.cell = rnn.MultiRNNCell(cells,state_is_tuple=True)
        init_state = self.cell.zero_state(self.args.batch_size,tf.float32)
        output_rnn, final_state = tf.nn.dynamic_rnn(self.cell, input_rnn, initial_state=init_state, dtype=tf.float32)
        output = tf.reshape(output_rnn,[-1,self.args.rnn_size])
        w_out = weights['out']
        b_out = biases['out']
        pred = tf.matmul(output, w_out) + b_out
        pred = tf.reshape(pred,[batch_size, time_step, self.args.output_size])
        return pred, final_state

    def SmoothL1(self,pred,Y,sigma):
        sigma2 = sigma * sigma
        num = tf.shape(Y)[0]
        diffs = tf.subtract(pred, Y)
        smooth_l1_sign = tf.cast(tf.less(tf.abs(diffs), 1.0/sigma), tf.float32)
        smooth_l1_opion1 = tf.multiply(tf.multiply(diffs, diffs), 0.5*sigma2)
        smooth_l1_opion2 = tf.subtract(tf.abs(diffs), 0.5/sigma2)

        smooth_l1_result = tf.add(tf.multiply(smooth_l1_opion1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_opion2, tf.subtract(1.0,smooth_l1_sign)))
        return smooth_l1_result
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import _zero_state_tensors
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from tensorflow.contrib.rnn import GRUCell
import numpy as np
import pandas as pd
import collections
import os
import json
import re
import collections
import datetime
import pickle
import random
from tqdm import trange
import time


class quora_question_model(object):
    def __init__(self, voc_size, target_size, input_len_max, lr, dev, sess, makedir=True):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        print("Create quora_question_model class...")
        print()

        self.voc_size = voc_size
        self.target_size = target_size
        self.input_len_max = input_len_max
        self.lr = lr
        self.sess = sess
        self.dev = dev
        self.makedir = makedir
        
        self._build_graph()
        self.sess.run(tf.global_variables_initializer())
        

    def _build_graph(self):
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(now)
        print("Build Graph...")
        print()
    
        self.xavier_init = tf.contrib.layers.xavier_initializer()
        
        self.embed_dim = 100
        self.state_dim = 100
        self.bi_state_dim = self.state_dim * 2
        self.feat_dim = self.bi_state_dim
        self.attend_dim = self.feat_dim
        self.context_dim = self.bi_state_dim * 4
        self.fc_dim = 250
        
        print("embed_dim : %d" % self.embed_dim)
        print("state_dim : %d" % self.state_dim)
        print("bi_state_dim : %d" % self.bi_state_dim)
        print("feat_dim : %d" % self.feat_dim)
        print("attend_dim : %d" % self.attend_dim)
        print("context_dim : %d" % self.context_dim)
        print("fc_dim : %d" % self.fc_dim)
        print()
        
        with tf.device(self.dev):
            with tf.variable_scope("input_placeholders"):
                self.enc_input1 = tf.placeholder(tf.int32, shape=[None, None], name="enc_input1")
                self.enc_seq_len1 = tf.placeholder(tf.int32, shape=[None, ], name="enc_seq_len1")
                self.enc_input2 = tf.placeholder(tf.int32, shape=[None, None], name="enc_input2")
                self.enc_seq_len2 = tf.placeholder(tf.int32, shape=[None, ], name="enc_seq_len2")
                self.targets = tf.placeholder(tf.int32, shape=[None, ], name="targets")
                self.batch_size = tf.placeholder(tf.int32, shape=[], name="batch_size")
                self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")

            with tf.variable_scope("words_embedding"):
                self.embeddings = tf.get_variable("embeddings", [self.voc_size, self.embed_dim], initializer=self.xavier_init)
                self.embed_in1 = tf.nn.embedding_lookup(self.embeddings, self.enc_input1, name="embed_in1")
                self.embed_in2 = tf.nn.embedding_lookup(self.embeddings, self.enc_input2, name="embed_in2")
                
                self.pad_mask1 = tf.sequence_mask(self.enc_seq_len1, self.input_len_max, dtype=tf.float32, name="pad_mask1")
                self.pad_mask2 = tf.sequence_mask(self.enc_seq_len2, self.input_len_max, dtype=tf.float32, name="pad_mask2")

            with tf.variable_scope("rnn_encoder_layer") as scope_rnn:
                self.output_enc1, self.state_enc1 = bi_rnn(GRUCell(self.state_dim), GRUCell(self.state_dim),
                                               inputs=self.embed_in1, sequence_length=self.enc_seq_len1, dtype=tf.float32)

                self.state_enc1 = tf.concat([self.state_enc1[0], self.state_enc1[1]], axis=1, name="state_enc1")
                assert self.state_enc1.get_shape()[1] == self.bi_state_dim

                self.output_enc1 = tf.concat(self.output_enc1, axis=2)  # [batch, max_eng, state*2]
                self.output_enc1 = tf.nn.dropout(self.output_enc1, keep_prob=self.keep_prob, name="output_enc1")
                print("output_enc1.get_shape() : %s" % (self.output_enc1.get_shape()))
                assert self.output_enc1.get_shape()[2] == self.bi_state_dim
                
                scope_rnn.reuse_variables()
                
                self.output_enc2, self.state_enc2 = bi_rnn(GRUCell(self.state_dim), GRUCell(self.state_dim),
                                               inputs=self.embed_in2, sequence_length=self.enc_seq_len2, dtype=tf.float32)

                self.state_enc2 = tf.concat([self.state_enc2[0], self.state_enc2[1]], axis=1, name="state_enc2")
                assert self.state_enc2.get_shape()[1] == self.bi_state_dim
                
                self.output_enc2 = tf.concat(self.output_enc2, axis=2)  # [batch, max_eng, state*2]
                self.output_enc2 = tf.nn.dropout(self.output_enc2, keep_prob=self.keep_prob, name="output_enc2")
                print("output_enc2.get_shape() : %s" % (self.output_enc2.get_shape()))
                assert self.output_enc2.get_shape()[2] == self.bi_state_dim
                
            with tf.variable_scope("attention_layer") as scope_attention:
                self.W_y = tf.get_variable("W_y", [1, 1, self.feat_dim, self.attend_dim], initializer=self.xavier_init)
                self.W_h = tf.get_variable("W_h", [self.feat_dim, self.attend_dim], initializer=self.xavier_init)
                self.W_a = tf.get_variable("W_a", [self.attend_dim, 1], initializer=self.xavier_init)

                # question 1..
                # average vector
                self.R_ave_1 = tf.reduce_mean(self.output_enc1, axis=1, name="R_ave_1")
                print("R_ave_1.get_shape() : %s" % (self.R_ave_1.get_shape()))
                
                # Wy * Y
                self.output_enc1_ex = tf.reshape(self.output_enc1, [-1, self.input_len_max, 1, self.feat_dim])
                self.M_1_left = tf.nn.conv2d(self.output_enc1_ex, self.W_y, strides=[1,1,1,1], padding="SAME")
                self.M_1_left = tf.reshape(self.M_1_left, [-1, self.input_len_max, self.attend_dim])
                print("M_1_left.get_shape() : %s" % (self.M_1_left.get_shape()))
                
                # Wh * Rave
                self.M_1_right = tf.matmul(self.R_ave_1, self.W_h)
                self.M_1_right = tf.ones([self.input_len_max, 1, 1]) * self.M_1_right
                self.M_1_right = tf.transpose(self.M_1_right, [1, 0, 2])
                print("M_1_right.get_shape() : %s" % (self.M_1_right.get_shape()))
                
                # attention
                self.M_1 = tf.tanh(self.M_1_left + self.M_1_right)
                print("M_1.get_shape() : %s" % (self.M_1.get_shape()))
                
                self.w_M_1 = tf.matmul(tf.reshape(self.M_1, [-1, self.attend_dim]), self.W_a)
                self.w_M_1 = tf.reshape(self.w_M_1, [-1, self.input_len_max])
                print("w_M_1.get_shape() : %s" % (self.w_M_1.get_shape()))
                
                self.attention1 = tf.nn.softmax(self.w_M_1) * self.pad_mask1
                self.attention1 = self.attention1 / tf.reshape(tf.reduce_sum(self.attention1, axis=1), [-1, 1])
                print("attention1.get_shape() : %s" % (self.attention1.get_shape()))

                self.context1 = tf.reduce_sum(self.output_enc1 *
                                              tf.reshape(self.attention1, [-1, self.input_len_max, 1]), 
                                              axis=1, 
                                              name="context1")
                print("context1.get_shape() : %s" % (self.context1.get_shape()))
                
                # question 2..
                # average vector
                self.R_ave_2 = tf.reduce_mean(self.output_enc2, axis=1, name="R_ave_2")
                print("R_ave_2.get_shape() : %s" % (self.R_ave_2.get_shape()))
                
                # Wy * Y
                self.output_enc2_ex = tf.reshape(self.output_enc2, [-1, self.input_len_max, 1, self.feat_dim])
                self.M_2_left = tf.nn.conv2d(self.output_enc2_ex, self.W_y, strides=[1,1,1,1], padding="SAME")
                self.M_2_left = tf.reshape(self.M_2_left, [-1, self.input_len_max, self.attend_dim])                 
                print("M_2_left.get_shape() : %s" % (self.M_2_left.get_shape()))
                
                # Wh * Rave
                self.M_2_right = tf.matmul(self.R_ave_2, self.W_h)
                self.M_2_right = tf.ones([self.input_len_max, 1, 1]) * self.M_2_right
                self.M_2_right = tf.transpose(self.M_2_right, [1, 0, 2])
                print("M_2_right.get_shape() : %s" % (self.M_2_right.get_shape()))
                
                # attention
                self.M_2 = tf.tanh(self.M_2_left + self.M_2_right)
                print("M_2.get_shape() : %s" % (self.M_2.get_shape()))
                
                self.w_M_2 = tf.matmul(tf.reshape(self.M_2, [-1, self.attend_dim]), self.W_a)
                self.w_M_2 = tf.reshape(self.w_M_2, [-1, self.input_len_max])
                print("w_M_2.get_shape() : %s" % (self.w_M_2.get_shape()))

                self.attention2 = tf.nn.softmax(self.w_M_2) * self.pad_mask2
                self.attention2 = self.attention2 / tf.reshape(tf.reduce_sum(self.attention2, axis=1), [-1, 1])
                print("attention2.get_shape() : %s" % (self.attention2.get_shape()))
                
                self.context2 = tf.reduce_sum(self.output_enc2 *
                                              tf.reshape(self.attention2, [-1, self.input_len_max, 1]), 
                                              axis=1, 
                                              name="context2")
                print("context2.get_shape() : %s" % (self.context2.get_shape()))
                
                assert self.context1.get_shape()[1] == self.feat_dim
                assert self.context2.get_shape()[1] == self.feat_dim
            
            with tf.variable_scope("final_context_layer"):
                self.features = [self.context1, 
                                 self.context2, 
                                 tf.abs(self.context1 - self.context2), 
                                 (self.context1 * self.context2)]
                self.merged_feature = tf.concat(self.features, axis=1, name="merged_feature")
                print("merged_feature.get_shape() : %s" % (self.merged_feature.get_shape()))
                assert self.merged_feature.get_shape()[1] == self.context_dim
                
            with tf.variable_scope("dense_layer"):
                self.W_out1 = tf.get_variable("W_out1", [self.context_dim, self.fc_dim], initializer=self.xavier_init)
                self.bias_out1 = tf.get_variable("bias_out1", [self.fc_dim])
                self.W_out2 = tf.get_variable("W_out2", [self.fc_dim, self.target_size], initializer=self.xavier_init)
                self.bias_out2 = tf.get_variable("bias_out2", [self.target_size])

                self.fc = tf.nn.xw_plus_b(self.merged_feature, self.W_out1, self.bias_out1)
                self.fc = tf.tanh(self.fc)
                print("fc.get_shape() : %s" % (self.fc.get_shape()))
                
                self.y_hat = tf.nn.xw_plus_b(self.fc, self.W_out2, self.bias_out2, name="y_hat")
                print("y_hat.get_shape() : %s" % (self.y_hat.get_shape()))
                
            with tf.variable_scope("train_optimization"):
                self.train_vars = tf.trainable_variables()
                
                print()
                print("trainable_variables")
                for varvar in self.train_vars:
                    print(varvar)
                print()
                
                self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.y_hat, labels=self.targets)
                self.loss = tf.reduce_mean(self.loss, name="loss")
                self.loss_l2 = tf.add_n([tf.nn.l2_loss(v) for v in self.train_vars if "bias" not in v.name]) * 0.0001
                self.loss = self.loss + self.loss_l2
                
                self.predict = tf.argmax(tf.nn.softmax(self.y_hat), 1)
                self.predict = tf.cast(tf.reshape(self.predict, [self.batch_size, 1]), tf.int32, name="predict")

                self.target_label = tf.cast(tf.reshape(self.targets, [self.batch_size, 1]), tf.int32)
                self.correct = tf.equal(self.predict, self.target_label)
                self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32))
                
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                self.decay_rate = tf.maximum(0.00007, 
                                             tf.train.exponential_decay(self.lr, self.global_step, 
                                                                        1500, 0.95, staircase=True), 
                                             name="decay_rate")
                self.opt = tf.train.AdamOptimizer(learning_rate=self.decay_rate)
                self.grads_and_vars = self.opt.compute_gradients(self.loss, self.train_vars)
                self.grads_and_vars = [(tf.clip_by_norm(g, 30.0), v) for g, v in self.grads_and_vars]
                self.grads_and_vars = [(tf.add(g, tf.random_normal(tf.shape(g), stddev=0.001)), v) for g, v in self.grads_and_vars]

                self.train_op = self.opt.apply_gradients(self.grads_and_vars, global_step=self.global_step, name="train_op")
            
            if self.makedir == True:
                # Summaries for loss and lr
                self.loss_summary = tf.summary.scalar("loss", self.loss)
                self.accuracy_summary = tf.summary.scalar("accuracy", self.accuracy)
                self.lr_summary = tf.summary.scalar("lr", self.decay_rate)

                # Output directory for models and summaries
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                self.out_dir = os.path.abspath(os.path.join("./model", timestamp))
                print("LOGDIR = %s" % self.out_dir)
                print()

                # Train Summaries
                self.train_summary_op = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.lr_summary])
                self.train_summary_dir = os.path.join(self.out_dir, "summary", "train")
                self.train_summary_writer = tf.summary.FileWriter(self.train_summary_dir, self.sess.graph)

                # Test summaries
                self.test_summary_op = tf.summary.merge([self.loss_summary, self.accuracy_summary, self.lr_summary])
                print(self.test_summary_op)
                self.test_summary_dir = os.path.join(self.out_dir, "summary", "test")
                self.test_summary_writer = tf.summary.FileWriter(self.test_summary_dir, self.sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
                self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model-step")
                if not os.path.exists(self.checkpoint_dir):
                    os.makedirs(self.checkpoint_dir)

            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=None)

            
    def batch_train(self, batchs, data_x1, data_x2, data_y, len_x1, len_x2, writer=False):
        feed_dict = {self.enc_input1: data_x1, 
                     self.enc_seq_len1: len_x1,
                     self.enc_input2: data_x2, 
                     self.enc_seq_len2: len_x2,
                     self.targets: data_y,
                     self.batch_size: batchs,
                     self.keep_prob: 0.75}

        if writer == True:
            results = \
            self.sess.run([self.train_op, self.predict, self.loss, self.accuracy, 
                           self.attention1, self.attention2, 
                           self.global_step, self.decay_rate, self.train_summary_op], 
                          feed_dict)
            
            ret = [results[1], results[2], results[3], results[4], results[5], results[6], results[7]]
        
            self.train_summary_writer.add_summary(results[8], results[6])
        else:
            results = \
            self.sess.run([self.train_op, self.predict, self.loss, self.accuracy, 
                           self.attention1, self.attention2, self.global_step, self.decay_rate], 
                          feed_dict)
            
            ret = [results[1], results[2], results[3], results[4], results[5], results[6], results[7]]
        
        return ret

    
    def batch_test(self, batchs, data_x1, data_x2, data_y, len_x1, len_x2, writer=False):
        feed_dict = {self.enc_input1: data_x1, 
                     self.enc_seq_len1: len_x1,
                     self.enc_input2: data_x2, 
                     self.enc_seq_len2: len_x2,
                     self.targets: data_y,
                     self.batch_size: batchs,
                     self.keep_prob: 1.0}
        
        if writer == True:
            results = \
                self.sess.run([self.predict, self.loss, self.accuracy, self.attention1, self.attention2, 
                               self.global_step, self.decay_rate, self.test_summary_op], 
                              feed_dict)

            ret = [results[0], results[1], results[2], results[3], results[4], results[5], results[6]]

            self.test_summary_writer.add_summary(results[7], results[5])
        else:
            results = \
                self.sess.run([self.predict, self.loss, self.accuracy, self.attention1, self.attention2, 
                               self.global_step, self.decay_rate], 
                              feed_dict)

            ret = [results[0], results[1], results[2], results[3], results[4], results[5], results[6]]
            
        return ret
    
    
    def save_model(self):
        current_step = tf.train.global_step(self.sess, self.global_step)
        self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
        
        
    def load_model(self, file_model):
        print("Load model (%s)..." % file_model)
        #file_model = "./model/2017-12-20 11:19/checkpoints/"
        #self.saver.restore(self.sess, tf.train.latest_checkpoint(file_model))
        self.saver.restore(self.sess, file_model)


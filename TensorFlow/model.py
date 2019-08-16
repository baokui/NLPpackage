# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 18:23:53 2019

@author: 12
"""

import tensorflow as tf
import modules
def model_mnist_simple(config):
    batch_size = config.batch_size
    with tf.name_scope('inputs'):
        X_holder = tf.placeholder(tf.float32)
        y_holder = tf.placeholder(tf.float32)
    with tf.name_scope('parameters'):
        Weights = tf.Variable(tf.zeros([784, 10]))
        biases = tf.Variable(tf.zeros([1,10]))
    with tf.name_scope('outputs_loss'):
        predict_y = tf.nn.softmax(tf.matmul(X_holder, Weights) + biases)
        loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
    with tf.name_scope('train_op'):
        optimizer = modules.create_optimizer(init_learning_rate=config.init_learning_rate,
                                       end_learning_rate=config.end_learning_rate,
                                       warmup_steps=int(config.epochs * config.nb_examples * 0.1 / config.batch_size),
                                       decay_steps=int(config.epochs * config.nb_examples * 0.9 / config.batch_size))
        optimizer = tf.train.GradientDescentOptimizer(0.5)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.clip_grad)
        train_op = optimizer.minimize(loss)
        #train_op = optimizer.apply_gradients(zip(grads, tvars))
        return X_holder, y_holder, predict_y, loss, optimizer, train_op,grads
        # global_step = tf.train.get_or_create_global_step()
        # train_op = tf.group(optOp, [tf.assign_add(global_step, 1)])
        # optimizer = tf.train.GradientDescentOptimizer(0.5)
        # train = optimizer.minimize(loss)

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
        #grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), config.clip_grad)
        grads = optimizer.compute_gradients(loss,var_list=tf.get_collection(
                                          tf.GraphKeys.TRAINABLE_VARIABLES,
                                          scope='parameters'
                                      ))
        train_op = optimizer.minimize(loss)
        #train_op = optimizer.apply_gradients(zip(grads, tvars))
        correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return X_holder, y_holder, predict_y, loss, optimizer, train_op,grads,accuracy
        # global_step = tf.train.get_or_create_global_step()
        # train_op = tf.group(optOp, [tf.assign_add(global_step, 1)])
        # optimizer = tf.train.GradientDescentOptimizer(0.5)
        # train = optimizer.minimize(loss)
def model_mnist_simple_mutiGPU(config):
    def average_gradients(tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expend_g = tf.expand_dims(g, 0)
                grads.append(expend_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads
    batch_size = config.batch_size
    with tf.name_scope('inputs'):
        X_holder = tf.placeholder(tf.float32,shape=[None,784])
        y_holder = tf.placeholder(tf.float32,shape=[None,10])
    with tf.name_scope('parameters'):
        Weights = tf.Variable(tf.zeros([784, 10]))
        biases = tf.Variable(tf.zeros([1,10]))
    #global_step = tf.train.get_or_create_global_step()
    p = []
    tower_grads = []
    tower_input_loss = []
    sn_op = []
    optimizer = tf.train.GradientDescentOptimizer(0.5)
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(len(config.GPU.split(','))):
            gpu = config.GPU.split(',')[i]
            with tf.device(gpu):
                with tf.name_scope('tower_' + gpu[-1]):
                    predict_y = tf.nn.softmax(
                        tf.matmul(X_holder[i * config.batch_size:(i + 1) * config.batch_size], Weights) + biases)
                    loss = tf.reduce_mean(
                        -tf.reduce_sum(y_holder[i * config.batch_size:(i + 1) * config.batch_size] * tf.log(predict_y),
                                       1))
                    grads = optimizer.compute_gradients(loss, var_list=tf.get_collection(
                        tf.GraphKeys.TRAINABLE_VARIABLES,
                        scope='parameters'
                    ))
                    train_op = optimizer.minimize(loss)
                    tower_grads.append(grads)
                    tower_input_loss.append(loss)
                    sn_op.append(train_op)
                    p.append(predict_y)
    p = tf.concat(p, axis=0)
    # grads = modules.average_gradients(tower_grads)
    input_loss = tf.reduce_mean(tower_input_loss)

    # train_op = optimizer.apply_gradients(grads)
    # #train_op = tf.group(train_op, [tf.assign_add(global_step, 1)])
    # train_op = tf.group(train_op, sn_op)

    grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(grads)

    predict_test = tf.nn.softmax(
        tf.matmul(X_holder, Weights) + biases)
    correct_prediction = tf.equal(tf.argmax(predict_test, 1), tf.argmax(y_holder, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return X_holder, y_holder, p, input_loss, optimizer, train_op,grads,accuracy
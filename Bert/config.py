# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:18:56 2019

@author: guo bk
"""

import copy
import json
import tensorflow as tf

class Config_bert(object):
    def __init__(self,
                 vocab_size=1000,
                 epochs=40,
                 examples=74092,
                 num_sampled = 64,
                 batch_size=32,
                 nb_context=4,
                 steps_per_checkpoint=1000,
                 training_steps=1000000,
                 input_maxlen=16,
                 CKPT_path="./CKPT/",
                 LOG_path="./LOG/",
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 attention_mask=True,
                 init_learning_rate=1e-4,
                 end_learning_rate=1e-6,
                 dropout_prob=0.1,
                 num_attention_heads=5,
                 size_per_head=60,
                 query_act=None,
                 key_act=None,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02,
                 intermediate_size=1024,
                 hidden_size=300,
                 dim_latent=128,
                 data_dir="/home/Research/NLP-Chinese-Corpus/NLP-Chinese-Corpus/ERNIE_task_data/lcqmc/",
                 bert_config_file="",
                 vocab_file="",
                 init_checkpoint="",
                 do_lower_case=True,
                 max_seq_length=128,
                 do_train=True,
                 do_eval=False,
                 do_predict=False,
                 train_batch_size=32,
                 eval_batch_size=32,
                 predict_batch_size=32,
                 learning_rate=5e-5,
                 num_train_epochs=3,
                 warmup_proportion=0.1,
                 save_checkpoints_steps=1000,
                 iterations_per_loop=1000,
                 use_tpu=False
                 ):
        self.vocab_size = vocab_size
        self.epochs = epochs
        self.examples = examples
        self.num_sampled = num_sampled
        self.nb_context = nb_context
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.attention_mask = attention_mask
        self.init_learning_rate = init_learning_rate
        self.end_learning_rate = end_learning_rate
        self.dropout_prob = dropout_prob
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.query_act = query_act
        self.key_act = key_act
        self.value_act = value_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.steps_per_checkpoint = steps_per_checkpoint
        self.input_maxlen = input_maxlen
        self.CKPT_path = CKPT_path
        self.LOG_path = LOG_path
        self.dim_latent = dim_latent
        self.training_steps = training_steps
        self.data_dir = data_dir
        self.bert_config_file = bert_config_file
        self.vocab_file = vocab_file
        self.init_checkpoint=init_checkpoint
        self.do_lower_case=do_lower_case
        self.max_seq_length=max_seq_length
        self.do_train=do_train
        self.do_eval=do_eval
        self.do_predict=do_predict
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.predict_batch_size = predict_batch_size
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.warmup_proportion = warmup_proportion
        self.save_checkpoints_steps = save_checkpoints_steps
        self.iterations_per_loop = iterations_per_loop
        self.use_tpu = use_tpu
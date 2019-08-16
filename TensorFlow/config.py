# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:18:56 2019

@author: guo bk
"""

import copy
import json
import tensorflow as tf
from modules import gelu
class Config(object):
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
                 intermediate_activation=gelu,
                 vocabfile='/home/GuoBaoKui/trio_work/bert/BERT-keras-master/train_baiduzhidao/vocab.txt',
                 word2vec='/home/Research/Word2vector/getvector/baiduzhidao_question_wiki300.npy',
                 sentencefiles=['/home/GuoBaoKui/trio_work/bert/BERT-keras-master/train_baiduzhidao/data/baidu_question_seg.txt']

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
        self.intermediate_activation = intermediate_activation
        self.batch_size = batch_size
        self.steps_per_checkpoint = steps_per_checkpoint
        self.input_maxlen = input_maxlen
        self.CKPT_path = CKPT_path
        self.LOG_path = LOG_path
        self.dim_latent = dim_latent
        self.training_steps = training_steps
        self.vocabfile = vocabfile
        self.sentencefiles = sentencefiles
        self.word2vec = word2vec

    @classmethod
    def from_dict(cls, dictionary):
        config = Config(vocab_size=None)
        for k, v in dictionary.items():
            config.__dict__[k] = v
        return config

    @classmethod
    def from_json_file(cls, json_file):
        with tf.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class Config_mnist(object):
    def __init__(self,
                 epochs=40,
                 CPU='/cpu:0',
                 GPU='/gpu:0,/gpu:1,/gpu:2,/gpu:3',
                 nb_examples=None,
                 batch_size=32,
                 steps_per_checkpoint=1000,
                 training_steps=1000000,
                 CKPT_path="./result_mnist/CKPT/",
                 LOG_path="./result_mnist/LOG/",
                 init_learning_rate=1e-4,
                 end_learning_rate=1e-6,
                 dropout_prob=0.1,
                 initializer_range=0.02,
                 clip_grad=0.1
                 ):
        self.epochs = epochs
        self.CPU = CPU
        self.GPU = GPU
        self.nb_examples = nb_examples
        self.init_learning_rate = init_learning_rate
        self.end_learning_rate = end_learning_rate
        self.dropout_prob = dropout_prob
        self.initializer_range = initializer_range
        self.batch_size = batch_size
        self.steps_per_checkpoint = steps_per_checkpoint
        self.CKPT_path = CKPT_path
        self.LOG_path = LOG_path
        self.training_steps = training_steps
        self.clip_grad = clip_grad


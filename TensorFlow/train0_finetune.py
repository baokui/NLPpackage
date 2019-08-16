# -*- coding: utf-8 -*-
"""
Created on Fri May 24 17:03:40 2019

@author: guo bk
"""
import modules
import tensorflow as tf
from config0 import Config
import math
import tokenizer as tknz
import os
import numpy as np
def build_encoder_fn(config,
                                    name='encoder'):
    '''
    Here adpots Transformer encoder structure,
    layer number is denoted in config as num_context_representation_layers
    '''
    def context_representation_fn(tensor):
        '''
        tensor: shape = [B, F, H]
        return: shape = [B, F, H]
        '''
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i in range(config.num_encoder_layers):
                tensor = modules.encoder_layer(
                    from_tensor=tensor,
                    to_tensor=tensor,
                    dropout_prob=config.dropout_prob,
                    num_attention_heads=config.num_attention_heads,
                    size_per_head=config.size_per_head,
                    hidden_size = config.hidden_size,
                    query_act=config.query_act,
                    key_act=config.key_act,
                    value_act=config.value_act,
                    attention_probs_dropout_prob=config.attention_probs_dropout_prob,
                    initializer_range=config.initializer_range,
                    intermediate_size=config.intermediate_size,
                    intermediate_activation=config.intermediate_activation,
                    name='encoder_%d' % i
                )
        return tensor
    return context_representation_fn
def debugbatch(files, tokenizer, input_max_length, batch_size,nb_context,min_length=5):
    n = int(nb_context/2)
    doc = []
    word_context = []
    word_target = []
    for file_index in range(len(files)):
        file = files[file_index]
        f = tf.gfile.GFile(file, 'r')
        while True:
            sentence = f.readline()
            if not sentence:
                continue
            sentence = sentence.strip()
            token_sentence = tokenizer.tokenize(sentence)
            if len(token_sentence)<min(nb_context,min_length):
                continue
            for i in range(n,len(token_sentence)-n):
                if np.random.random() < -1:
                    continue
                token_context = token_sentence[i-n:i]+token_sentence[i+1:i+n+1]
                doc.append(tokenizer.convert_tokens_to_ids(token_sentence, input_max_length))
                word_context.append(tokenizer.convert_tokens_to_ids(token_context, nb_context))
                word_target.append(tokenizer.convert_tokens_to_ids(token_sentence[i], 1))
            if len(doc)<batch_size:
                continue
            return doc[:batch_size],word_context[:batch_size],word_target[:batch_size]
def Iterator(files, tokenizer, input_max_length, batch_size,nb_context,min_length=5):
    n = int(nb_context/2)
    doc = []
    word_context = []
    word_target = []
    for file_index in range(len(files)):
        file = files[file_index]
        f = tf.gfile.GFile(file, 'r')
        while True:
            sentence = f.readline()
            if not sentence:
                if file_index == len(files) - 1:
                    yield '__STOP__'
                break
            sentence = sentence.strip().split('\t')[:2]
            for s0 in sentence:
                token_sentence = tokenizer.tokenize(s0)
                if len(token_sentence)<min(nb_context,min_length):
                    continue
                for i in range(n,len(token_sentence)-n):
                    if np.random.random() < -1:
                        continue
                    token_context = token_sentence[i-n:i]+token_sentence[i+1:i+n+1]
                    doc.append(tokenizer.convert_tokens_to_ids(token_sentence, input_max_length))
                    word_context.append(tokenizer.convert_tokens_to_ids(token_context, nb_context))
                    word_target.append(tokenizer.convert_tokens_to_ids(token_sentence[i], 1))
                    if len(doc)==batch_size:
                        yield doc[:batch_size],word_context[:batch_size],word_target[:batch_size]
                        doc,word_context,word_target = [],[],[]
config = Config()
batch_size = config.batch_size
max_len_doc = config.input_maxlen
nb_context = config.nb_context
with open(config.vocabfile,'r') as f:
    vocab = f.read().split('\n')[:-1]
tokenizer = tknz.Tokenizer(vocab)
vocab = tokenizer.vocab

embedding_table = modules.get_embeding_matrix(config.word2vec,vocab)
config.hidden_size = embedding_table.shape[1]
with tf.name_scope('inputs'):
    doc_input = tf.placeholder(tf.int32, shape=[batch_size, max_len_doc],name = 'doc_input')
    word_context = tf.placeholder(tf.int32, shape=[batch_size, nb_context],name = 'word_context')
    word_target = tf.placeholder(tf.int32, shape=[batch_size,1],name = 'word_target')
with tf.name_scope('token_embedding'):
    embedding_table = modules.create_embedding_table(embedding_table,
                                             config.vocab_size,
                                             config.hidden_size,
                                             config.initializer_range)
    doc_emb = tf.nn.embedding_lookup(embedding_table, doc_input, name='doc_emb')
    context_emb = tf.nn.embedding_lookup(embedding_table, word_context, name='context_emb')
    context_emb_vector = tf.reduce_mean(context_emb,1)

    position_embedding_table = modules.create_position_embedding_table(max_len_doc,config.hidden_size)

    doc_emb_pos = modules.layer_norm(
            modules.add_position_embedding(doc_emb, position_embedding_table),
            name='doc_emb_pos')
with tf.name_scope('doc_encoder'):
    encoder_fn = build_encoder_fn(config,name='encoder')
    doc_context = encoder_fn(doc_emb_pos)
    doc_context_vector = tf.reduce_mean(doc_context,1)
with tf.name_scope('merge_doc_wordcontext'):
    merged_vector = tf.add(context_emb_vector,doc_context_vector,name = 'merged_vector')
with tf.name_scope('last_layer_para'):
    nce_weights = tf.Variable(
            tf.truncated_normal(
                    [config.vocab_size, config.hidden_size],
                    stddev=1.0 / math.sqrt(config.hidden_size)),
            name = 'last_weights')
    nce_biases = tf.Variable(
            tf.truncated_normal(
                    [config.vocab_size],
                    stddev=1.0 / math.sqrt(config.hidden_size)),
            name = 'last_bias')
with tf.name_scope('loss'):
    loss = tf.reduce_mean(
        tf.nn.nce_loss(
            weights=nce_weights,
            biases=nce_biases,
            labels=word_target,
            inputs=merged_vector,
            num_sampled=config.num_sampled,
            num_classes=config.hidden_size))
with tf.name_scope('train_op'):
    opt = modules.create_optimizer(init_learning_rate=config.init_learning_rate,
                               end_learning_rate=config.end_learning_rate,
                               warmup_steps=int(config.epochs * config.examples * 0.1 / config.batch_size),
                               decay_steps=int(config.epochs * config.examples * 0.9 / config.batch_size))
    optOp = opt.minimize(loss)
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.group(optOp, [tf.assign_add(global_step, 1)])
saver = tf.train.Saver()
tf.summary.scalar("loss",loss)
#for var in tf.trainable_variables():
    #tf.summary.histogram(var.name, var)
# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('\nNumber of paras:',np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))
    ckpt_file = tf.train.latest_checkpoint(config.CKPT_path)
    if ckpt_file:
        saver.restore(sess, ckpt_file)
    #tf.get_default_graph().finalize()
    train_writer = tf.summary.FileWriter(config.LOG_path,sess.graph)
    #batchiter = Iterator(config.sentencefiles, tokenizer, max_len_doc, config.batch_size,config.nb_context,min_length=5)
    #for i in range(1,config.training_steps+1):
    i = 0
    for epoch in range(config.epochs):
        batchiter = Iterator(config.finetunefiles, tokenizer, max_len_doc, config.batch_size,config.nb_context,min_length=5)
        while True:
            next_batch = next(batchiter)
            if next_batch is '__STOP__':
                break
            batch_doc,batch_word_context,batch_word_target = next_batch
            feed_dict={doc_input:batch_doc,word_context:batch_word_context,word_target:batch_word_target}
            if i % config.steps_per_checkpoint == 0:
                saver.save(sess,os.path.join(config.CKPT_path_finetune,'model.ckpt'),global_step=global_step) 
                _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict = feed_dict)
                print('After %d epochs and %d training step(s), loss, on training batch is %g.' % (epoch, step, loss_value))
                i = i+1
            else:
                _,loss_value,step,summary=sess.run([train_op,loss,global_step,merged_summary_op],feed_dict=feed_dict)
                train_writer.add_summary(summary, i)
                if i%100==0:
                    print('After %d epochs and %d training step(s), loss, on training batch is %g.' % (epoch, step, loss_value))
                i = i+1
    train_writer.close()
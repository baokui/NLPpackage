# coding: utf-8

import math

import numpy as np
import tensorflow as tf

def get_embeding_matrix(vector_path,vocab,dim=300):
    if vector_path[-3:]=='npy':
        embeding_table = np.load(vector_path)
        return embeding_table
    with open(vector_path,'r') as f:
        s = f.read()
    s = s.split('\n')[1:-1]
    s = [ss.split() for ss in s]
    V = [ss[0] for ss in s]
    embeding_table = np.zeros((len(vocab),dim))
    for v in vocab:
        if v in V:
            vector = s[V.index(v)][1:]
            vector = np.array([np.float(t) for t in vector])
        else:
            vector = np.random.randn(1,dim)[0]
        embeding_table[vocab[v]] = vector
    #np.save('data/embeding_table.npy',embeding_table)
    return embeding_table

def dropout(input_tensor, dropout_prob):
    if dropout_prob is None or dropout_prob == 0.0:
        return input_tensor

    output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
    return output


def create_initializer(initializer_range=0.02):
    return tf.truncated_normal_initializer(stddev=initializer_range)


def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        (np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))
    return x * cdf


def layer_norm(input_tensor, name=None):
    return tf.contrib.layers.layer_norm(
        inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)


def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=8,
                    size_per_head=64,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02):
    '''
    from tensor : [B, F, hidden size]
    to tensor : [B, T, hidden size]
    output : [B, F, H*N]
    '''

    from_shape = get_shape(from_tensor)
    to_shape = get_shape(to_tensor)

    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]

    # Scalar dimensions referenced here:
    #   B = batch size (number of sequences)
    #   F = `from_tensor` sequence length
    #   T = `to_tensor` sequence length
    #   N = `num_attention_heads`
    #   H = `size_per_head`

    # [B, F, H*N]
    query_layer = tf.layers.dense(
        from_tensor,
        num_attention_heads * size_per_head,
        activation=query_act,
        name="query",
        kernel_initializer=create_initializer(initializer_range))

    # [B, T, H*N]
    key_layer = tf.layers.dense(
        to_tensor,
        num_attention_heads * size_per_head,
        activation=key_act,
        name="key",
        kernel_initializer=create_initializer(initializer_range))

    # [B, T, H*N]
    value_layer = tf.layers.dense(
        to_tensor,
        num_attention_heads * size_per_head,
        activation=value_act,
        name="value",
        kernel_initializer=create_initializer(initializer_range))

    # [B, F, N, H]
    query_layer_split = tf.reshape(
        query_layer,
        shape=[-1, from_seq_length, num_attention_heads, size_per_head])

    # [B, N, F, H]
    query = tf.transpose(query_layer_split, perm=[0, 2, 1, 3], name='query_trans')

    # [B, T, N, H]
    key_layer_split = tf.reshape(
        key_layer,
        shape=[-1, to_seq_length, num_attention_heads, size_per_head])

    # [B, N, T, H]
    key = tf.transpose(key_layer_split, perm=[0, 2, 1, 3], name='key_trans')

    # [B, T, N, H]
    value_layer_split = tf.reshape(
        value_layer,
        shape=[-1, to_seq_length, num_attention_heads, size_per_head])

    # [B, N, T, H]
    value = tf.transpose(value_layer_split, perm=[0, 2, 1, 3], name='value_trans')

    # [B, N, F, T]
    attention_scores = tf.matmul(query, key, transpose_b=True)
    attention_scores = tf.multiply(attention_scores,
                                   1.0 / math.sqrt(float(size_per_head)))

    if attention_mask is not None:
        # `attention_mask` = [1, 1, F, T]
        attention_mask = tf.expand_dims(
            tf.expand_dims(
                create_attention_mask(from_seq_length, to_seq_length),
                axis=0),
            axis=0)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_scores += adder

    # Normalize the attention scores to probabilities.
    # [B, N, F, T]
    attention_probs = tf.nn.softmax(attention_scores)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

    # [B, N, F, H]
    context_layer = tf.matmul(attention_probs, value)

    # [B, F, N, H]
    context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

    # [B, F, N*H]
    context = tf.reshape(
        context_layer,
        shape=[-1, from_seq_length, num_attention_heads * size_per_head])

    return context


def attention_res_ln(from_tensor,
                     to_tensor,
                     attention_mask=None,
                     dropout_prob=0.1,
                     num_attention_heads=8,
                     size_per_head=64,
                     hidden_size = 512,
                     query_act=None,
                     key_act=None,
                     value_act=None,
                     attention_probs_dropout_prob=0.0,
                     initializer_range=0.02,
                     name='attention',
                     use_dense = False):
    with tf.variable_scope(name):
        context = attention_layer(from_tensor=from_tensor,
                                  to_tensor=to_tensor,
                                  attention_mask=attention_mask,
                                  num_attention_heads=num_attention_heads,
                                  size_per_head=size_per_head,
                                  query_act=query_act,
                                  key_act=key_act,
                                  value_act=value_act,
                                  attention_probs_dropout_prob=attention_probs_dropout_prob,
                                  initializer_range=initializer_range)
        if use_dense:
            context = tf.layers.dense(context,hidden_size)
        context = dropout(context, dropout_prob)
        output = layer_norm(from_tensor + context)
    return output


def feedforward_layer(input_tensor,
                      intermediate_size=1024,
                      intermediate_activation=gelu,
                      initializer_range=0.02):
    '''
    input_tensor: [B, F, H]
    output_tensor ; [B, F, H]
    '''
    shape = get_shape(input_tensor)
    hidden_size = shape[-1]

    intermediate_layer = tf.layers.dense(input_tensor,
                                         units=intermediate_size,
                                         activation=intermediate_activation,
                                         kernel_initializer=create_initializer(initializer_range),
                                         bias_initializer=tf.zeros_initializer(),
                                         name='intermediate_layer')
    output_layer = tf.layers.dense(intermediate_layer,
                                   units=hidden_size,
                                   activation=None,
                                   kernel_initializer=create_initializer(initializer_range),
                                   bias_initializer=tf.zeros_initializer(),
                                   name='output_layer')
    return output_layer


def ffd_res_ln(input_tensor,
               dropout_prob=0.1,
               intermediate_size=1024,
               intermediate_activation=gelu,
               initializer_range=0.02,
               name='FFD'):
    with tf.variable_scope(name):
        context = feedforward_layer(input_tensor,
                                    intermediate_size=intermediate_size,
                                    intermediate_activation=intermediate_activation,
                                    initializer_range=initializer_range)
        context = dropout(context, dropout_prob)
        output = layer_norm(input_tensor + context)
    return output


def encoder_layer(from_tensor,
                  to_tensor,
                  dropout_prob=0.1,
                  num_attention_heads=8,
                  size_per_head=64,
                  query_act=None,
                  key_act=None,
                  value_act=None,
                  hidden_size = 512,
                  attention_probs_dropout_prob=0.0,
                  initializer_range=0.02,
                  intermediate_size=1024,
                  intermediate_activation=gelu,
                  name='encoder',
                  use_dense = False):
    with tf.variable_scope(name):
        context = attention_res_ln(from_tensor,
                                   to_tensor,
                                   dropout_prob=dropout_prob,
                                   num_attention_heads=num_attention_heads,
                                   size_per_head=size_per_head,
                                   hidden_size = hidden_size,
                                   query_act=query_act,
                                   key_act=key_act,
                                   value_act=value_act,
                                   attention_probs_dropout_prob=attention_probs_dropout_prob,
                                   initializer_range=initializer_range,
                                   use_dense = use_dense)
        output = ffd_res_ln(context,
                            dropout_prob=dropout_prob,
                            intermediate_size=intermediate_size,
                            intermediate_activation=intermediate_activation,
                            initializer_range=initializer_range)
    return output


def decoder_layer(from_tensor,
                  to_tensor,
                  attention_mask=True,
                  dropout_prob=0.1,
                  num_attention_heads=8,
                  size_per_head=64,
                  query_act=None,
                  key_act=None,
                  value_act=None,
                  hidden_size = 512,
                  attention_probs_dropout_prob=0.0,
                  initializer_range=0.02,
                  intermediate_size=1024,
                  intermediate_activation=gelu,
                  name='decoder',
                  use_dense = False):
    with tf.variable_scope(name):
        mask_context = attention_res_ln(from_tensor=from_tensor,
                                        to_tensor=from_tensor,
                                        attention_mask=attention_mask,
                                        dropout_prob=dropout_prob,
                                        num_attention_heads=num_attention_heads,
                                        size_per_head=size_per_head,
                                        hidden_size = hidden_size,
                                        query_act=query_act,
                                        key_act=key_act,
                                        value_act=value_act,
                                        attention_probs_dropout_prob=attention_probs_dropout_prob,
                                        initializer_range=initializer_range,
                                        name='mask_attetion',
                                        use_dense = use_dense)
        context = attention_res_ln(from_tensor=mask_context,
                                   to_tensor=to_tensor,
                                   dropout_prob=dropout_prob,
                                   num_attention_heads=num_attention_heads,
                                   size_per_head=size_per_head,
                                   hidden_size = hidden_size,
                                   query_act=query_act,
                                   key_act=key_act,
                                   value_act=value_act,
                                   attention_probs_dropout_prob=attention_probs_dropout_prob,
                                   initializer_range=initializer_range,
                                   name='dec_enc_attetion',
                                   use_dense = use_dense)
        output = ffd_res_ln(context,
                            dropout_prob=dropout_prob,
                            intermediate_size=intermediate_size,
                            intermediate_activation=intermediate_activation,
                            initializer_range=initializer_range)
    return output


def create_optimizer(init_learning_rate,
                     end_learning_rate,
                     warmup_steps,
                     decay_steps):
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.polynomial_decay(init_learning_rate,
                                              global_step,
                                              decay_steps,
                                              end_learning_rate)
    if warmup_steps:
        global_step_int = tf.cast(global_step, tf.int32)
        global_step_float = tf.cast(global_step, tf.float32)
        warmup_steps_int = tf.cast(warmup_steps, tf.int32)
        warmup_steps_float = tf.cast(warmup_steps, tf.float32)
        is_warmup = tf.cast(global_step_int < warmup_steps_int, dtype=tf.float32)
        learning_rate = is_warmup * global_step_float * init_learning_rate / warmup_steps_float \
                        + (1.0 - is_warmup) * learning_rate
    opt = tf.train.AdamOptimizer(learning_rate)
    return opt


def create_position_embedding_table(max_length,
                                    hidden_size):
    '''
    output : [max_length, hidden_size]
    '''

    def create_column(dimension,
                      max_length,
                      hidden_size):
        if dimension % 2 == 0:
            col = np.arange(max_length)
            col = col / np.power(10000, dimension / hidden_size)
            col = np.sin(col)
        else:
            col = np.arange(max_length)
            col = col / np.power(10000, (dimension - 1) / hidden_size)
            col = np.cos(col)
        col = np.expand_dims(col, axis=-1)
        return col

    table = np.array([])
    table = table.reshape([max_length, 0])
    for dimension in range(hidden_size):
        col = create_column(dimension,
                            max_length,
                            hidden_size)
        table = np.concatenate((table, col), axis=-1)
    table = tf.constant(table,
                        dtype=tf.float32,
                        name='position_embedding_table')
    return table


def add_position_embedding(tensor,
                           table,
                           name = "add_positionemb"):
    shape = get_shape(tensor)
    seq_length = shape[1]
    hidden_size = shape[-1]
    expand_table = tf.expand_dims(table, axis=0)
    output = tensor + tf.slice(
        expand_table,
        [0, 0, 0],
        [-1, seq_length, hidden_size],
        name=name
    )
    return output


def convert_tensor_to_single_vector(tensor,
                                    initializer_range=0.02,
                                    name='convert_to_single_vector'):
    '''
    tensor: [B, F, H]
    return: [B, 1, H]
    '''
    shape = get_shape(tensor)
    hidden_size = shape[-1]
    with tf.variable_scope(name):
        # [B, 1, H]
        tensor_mean = tf.reduce_mean(tensor, axis=1, keep_dims=True)
        # [B, 1, H]
        query = tf.layers.dense(
            tensor_mean,
            units=hidden_size,
            activation=tf.nn.tanh,
            use_bias=True,
            kernel_initializer=create_initializer(initializer_range)
        )
        # [B, 1, F]
        tensor_attention_score = tf.matmul(query, tensor, transpose_b=True)
        # [B, 1, F]
        tensor_attention_prob = tf.nn.softmax(tensor_attention_score, axis=-1)
        # [B, 1, H]
        tensor_final = layer_norm(tf.matmul(tensor_attention_prob, tensor) + tensor_mean)
    return tensor_final


def convert_2vectors_into_value(x_vector,
                                y_vector,
                                initializer_range,
                                additional_quadratic_vars=False,
                                name='2vec_to_value'):
    '''
    x_vector: shape = [None, 1, H]
    y_vector: shape = [None, 1, H]
    return: shape = [None]
    '''
    hidden_size = get_shape(x_vector)[-1]
    with tf.variable_scope(name):
        y_vector = tf.math.l2_normalize(y_vector, axis=-1)
        if additional_quadratic_vars:
            y_trans = tf.layers.dense(
                y_vector,
                units=hidden_size,
                activation=None,
                use_bias=False,
                kernel_initializer=create_initializer(initializer_range)
            )
        else:
            y_trans = y_vector
        value = tf.squeeze(
            tf.matmul(
                tf.math.l2_normalize(x_vector, axis=-1),
                y_trans,
                transpose_b=True
            ) + tf.layers.dense(
                x_vector,
                units=1,
                activation=None,
                use_bias=False,
                kernel_initializer=create_initializer(initializer_range)
            ) + tf.layers.dense(
                y_vector,
                units=1,
                activation=None,
                use_bias=False,
                kernel_initializer=create_initializer(initializer_range)
            ), axis=[1, 2]
        )
    return value

def create_attention_mask(from_seq_length, to_seq_length):
    '''
    Mask to_tensor, unmasked position as 1, masked position as 0
    output: [F, T]
    '''
    # [F]
    from_range = tf.range(from_seq_length)
    # [F, 1]
    from_range = tf.expand_dims(from_range, axis=-1)
    # [T]
    to_range = tf.range(to_seq_length)
    # [1, T]
    to_range = tf.expand_dims(to_range, axis=0)
    # [F, T]
    attention_mask = tf.cast(from_range >= to_range, tf.float32)
    return attention_mask

def create_embedding_table(embedding_table,
                           vocab_size,
                           hidden_size,
                           initializer_range,
                           name='embedding_table'):
    if embedding_table is None:
        embedding_table = tf.get_variable(name=name,
                                          shape=[vocab_size, hidden_size],
                                          dtype=tf.float32,
                                          initializer=create_initializer(initializer_range))
    else:
        embedding_table = tf.constant(value=embedding_table,
                                      dtype=tf.float32)
    return embedding_table

def create_embedding_table_trainable(embedding_table,
                           vocab_size,
                           hidden_size,
                           initializer_range,
                           name='embedding_table'):
    if embedding_table is None:
        embedding_table = tf.get_variable(name=name,
                                          shape=[vocab_size, hidden_size],
                                          dtype=tf.float32,
                                          initializer=create_initializer(initializer_range))
    else:
        embedding_table = tf.get_variable(name=name,
                                          dtype=tf.float32,
                                          initializer=embedding_table)
    return embedding_table

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

def get_shape(tensor):
    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape

    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape
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

if __name__ == '__main__':
    pass

import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import config
import numpy as np
def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, probabilities) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            correct_prediction = tf.equal(predictions, label_ids)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            output_spec = [total_loss,train_op,predictions,accuracy]
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                    labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                    "eval_accuracy": accuracy,
                    "eval_loss": loss,
                }

            eval_metrics = (metric_fn,
                            [per_example_loss, label_ids, logits, is_real_example])
            output_spec = [total_loss,eval_metrics]
        else:
            output_spec = [{"probabilities": probabilities}]
        return output_spec

    return model_fn


def main():
    Config = config.Config_bert()
    bert_config = modeling.BertConfig.from_json_file(Config.bert_config_file)
    train_examples = [i for i in range(1000000)]
    num_train_steps = int(
        len(train_examples) / Config.train_batch_size * Config.num_train_epochs)
    num_warmup_steps = int(num_train_steps * Config.warmup_proportion)
    model_fn = model_fn_builder(
        bert_config=bert_config,
        num_labels=2,
        init_checkpoint=Config.init_checkpoint,
        learning_rate=Config.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=Config.use_tpu,
        use_one_hot_embeddings=Config.use_tpu)
    features = {}
    features["input_ids"] = tf.placeholder(tf.int32,shape=[None,Config.max_seq_length])
    features["input_mask"] = tf.placeholder(tf.int32, shape=[None, Config.max_seq_length])
    features["segment_ids"] = tf.placeholder(tf.int32, shape=[None, Config.max_seq_length])
    features["label_ids"] = tf.placeholder(tf.int32, shape=[None])
    [loss,train_op,predictions,accuracy] = model_fn(features,None,'train',None)
# if 1:
#     x0 = np.random.randint(0, 100,size=[32,Config.max_seq_length])
#     x1 = np.random.randint(0, 1, size=[32, Config.max_seq_length])
#     x2 = np.random.randint(0, 1, size=[32, Config.max_seq_length])
#     y = np.random.randint(0,1,size=[32])
#     feed_dict={features["input_ids"]:x0,features["input_mask"]:x1,features["segment_ids"]:x2,features["label_ids"]:y}
#     sess = tf.Session()
#     init = tf.global_variables_initializer()
#     sess.run(init)
#     for i in range(1000):
#         [_loss,_] = sess.run([loss,train_op],feed_dict=feed_dict)
#         if i%10==0:
#             _acc = sess.run(accuracy,feed_dict=feed_dict)
#             print(_loss,_acc)



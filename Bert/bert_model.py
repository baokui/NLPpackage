import collections
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import config
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
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
            output_spec = [total_loss, train_op, predictions, accuracy]
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
            output_spec = [total_loss, eval_metrics]
        else:
            output_spec = [{"probabilities": probabilities}]
        return output_spec

    return model_fn


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        A = []
        B = []
        L = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[0])
                text_b = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[0])
                text_b = tokenization.convert_to_unicode(line[1])
                label = tokenization.convert_to_unicode(line[2])
            A.append(text_a)
            B.append(text_b)
            L.append(label)
        return A, B, L


def convert_single_example(text_a,text_b, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    tokens_a = tokenizer.tokenize(text_a)
    tokens_b = tokenizer.tokenize(text_b)

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)
    for token in tokens_b:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    input_ids, input_mask, segment_ids = input_ids[:max_seq_length],input_mask[:max_seq_length],segment_ids[:max_seq_length]
    return input_ids,input_mask,segment_ids
def iter_training(tokenizer,A,B,L,Config):
    input_ids = []
    input_mask = []
    segment_ids = []
    labels = []
    for i in range(len(A)):
        x0,x1,x2 = convert_single_example(A[i],B[i],Config.max_seq_length,tokenizer)
        y = int(L[i])
        input_ids.append(x0)
        input_mask.append(x1)
        segment_ids.append(x2)
        labels.append((y))
        if len(input_ids)==Config.train_batch_size:
            yield input_ids,input_mask,segment_ids,labels
            input_ids = []
            input_mask = []
            segment_ids = []
            labels = []
    yield '__end__'
def dev_data(tokenizer,A,B,L,Config):
    input_ids = []
    input_mask = []
    segment_ids = []
    labels = []
    for i in range(len(A)):
        x0,x1,x2 = convert_single_example(A[i],B[i],Config.max_seq_length,tokenizer)
        y = int(L[i])
        input_ids.append(x0)
        input_mask.append(x1)
        segment_ids.append(x2)
        labels.append((y))
    return input_ids,input_mask,segment_ids,labels
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
    features["input_ids"] = tf.placeholder(tf.int32, shape=[None, Config.max_seq_length])
    features["input_mask"] = tf.placeholder(tf.int32, shape=[None, Config.max_seq_length])
    features["segment_ids"] = tf.placeholder(tf.int32, shape=[None, Config.max_seq_length])
    features["label_ids"] = tf.placeholder(tf.int32, shape=[None])
    [loss, train_op, predictions, accuracy] = model_fn(features, None, 'train', None)
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.group(train_op, [tf.assign_add(global_step, 1)])
    pro = ColaProcessor()
    A,B,L = pro.get_train_examples(Config.data_dir)
    A = A[1:]
    B = B[1:]
    L = L[1:]
    At,Bt,Lt = pro.get_dev_examples(Config.data_dir)
    At = At[1:]
    Bt = Bt[1:]
    Lt = Lt[1:]
    tokenizer = tokenization.FullTokenizer(
        vocab_file=Config.vocab_file, do_lower_case=Config.do_lower_case)
    iter = iter_training(tokenizer, A, B, L, Config)
    X0,X1,X2,Y = dev_data(tokenizer, At, Bt, Lt, Config)
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=12)
    init = tf.global_variables_initializer()
    sess.run(init)
    epoch = 0
    i = 0
    j = 0
    while epoch<Config.epochs:
        data = next(iter)
        if data=='__end__':
            epoch += 1
            iter = iter_training(tokenizer, A, B, L, Config)
            continue
        [x0, x1, x2, y] = data
        feed_dict={features["input_ids"]:x0,features["input_mask"]:x1,features["segment_ids"]:x2,features["label_ids"]:y}
        [_loss, _] = sess.run([loss, train_op], feed_dict=feed_dict)
        if i % 100 == 0:
            x0 = X0[j*Config.eval_batch_size:(j+1)*Config.eval_batch_size]
            x1 = X1[j*Config.eval_batch_size:(j+1)*Config.eval_batch_size]
            x2 = X2[j*Config.eval_batch_size:(j+1)*Config.eval_batch_size]
            y = Y[j*Config.eval_batch_size:(j+1)*Config.eval_batch_size]
            feed_dict={features["input_ids"]:x0,features["input_mask"]:x1,features["segment_ids"]:x2,features["label_ids"]:y}
            j += 1
            if j*Config.eval_batch_size>=len(X0):
                j = 0
            _acc = sess.run(accuracy, feed_dict=feed_dict)
            print(epoch,i,_loss, _acc)
            saver.save(sess, 'CKPT/bert_lcqmc', global_step=global_step)
        i += 1
def evaluation():
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
    features["input_ids"] = tf.placeholder(tf.int32, shape=[None, Config.max_seq_length])
    features["input_mask"] = tf.placeholder(tf.int32, shape=[None, Config.max_seq_length])
    features["segment_ids"] = tf.placeholder(tf.int32, shape=[None, Config.max_seq_length])
    features["label_ids"] = tf.placeholder(tf.int32, shape=[None])
    [loss, train_op, predictions, accuracy] = model_fn(features, None, 'train', None)
    global_step = tf.train.get_or_create_global_step()
    train_op = tf.group(train_op, [tf.assign_add(global_step, 1)])
    pro = ColaProcessor()
    A, B, L = pro.get_train_examples(Config.data_dir)
    A = A[1:]
    B = B[1:]
    L = L[1:]
    At, Bt, Lt = pro.get_dev_examples(Config.data_dir)
    At = At[1:]
    Bt = Bt[1:]
    Lt = Lt[1:]
    tokenizer = tokenization.FullTokenizer(
        vocab_file=Config.vocab_file, do_lower_case=Config.do_lower_case)
    X0, X1, X2, Y = dev_data(tokenizer, At, Bt, Lt, Config)
    sess = tf.Session()
    saver = tf.train.Saver(max_to_keep=None, keep_checkpoint_every_n_hours=12)
    model_file = tf.train.latest_checkpoint('CKPT/')
    saver.restore(sess, model_file)
    j = 0
    r = []
    while True:
        x0 = X0[j * Config.eval_batch_size:(j + 1) * Config.eval_batch_size]
        x1 = X1[j * Config.eval_batch_size:(j + 1) * Config.eval_batch_size]
        x2 = X2[j * Config.eval_batch_size:(j + 1) * Config.eval_batch_size]
        y = Y[j * Config.eval_batch_size:(j + 1) * Config.eval_batch_size]
        feed_dict = {features["input_ids"]: x0, features["input_mask"]: x1, features["segment_ids"]: x2,
                     features["label_ids"]: y}
        j += 1
        if j * Config.eval_batch_size >= len(X0):
            break
        _acc = sess.run(accuracy, feed_dict=feed_dict)
        print(j,len(x0),_acc)
        r.append([len(x0),_acc])
    S = sum([rr[0] for rr in r])
    T = sum([rr[0]*rr[1] for rr in r])
    print(T/S)

if __name__=='__main__':
    import sys
    if len(sys.argv)==1:
        main()
    else:
        if sys.argv[1]=='train':
            main()
        else:
            evaluation()

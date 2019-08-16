import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import model_mnist_simple
import modules
import config as Config
import time
def test_singleGPU():
    config = Config.Config_mnist()
    mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)
    config.nb_examples = mnist.train.num_examples
    X_holder, y_holder, predict_y, loss, optimizer, train_op, _ = model_mnist_simple(config)
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    for i in range(500):
        images, labels = mnist.train.next_batch(config.batch_size)
        session.run(train_op, feed_dict={X_holder:images, y_holder:labels})
        if i % 25 == 0:
            correct_prediction = tf.equal(tf.argmax(predict_y, 1), tf.argmax(y_holder, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_value = session.run(accuracy, feed_dict={X_holder:mnist.test.images, y_holder:mnist.test.labels})
            print('step:%d accuracy:%.4f' %(i, accuracy_value))
def test_multiGPU():
    config = Config.Config_mnist()
    mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)
    config.nb_examples = mnist.train.num_examples
    X = []
    y = []
    p = []
    tower_grads = []
    tower_input_loss = []
    tower_input_logits = []
    sn_op = []
    opt = tf.train.AdamOptimizer(config.init_learning_rate, epsilon=1e-9)
    global_step = tf.train.get_or_create_global_step()
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        for i in range(len(config.GPU.split(','))):
            gpu = config.GPU.split(',')[i]
            with tf.device(gpu):
                with tf.name_scope('tower_' + gpu[-1]):
                    X_holder, y_holder, predict_y, loss, optimizer, train_op, grads = model_mnist_simple(config)
                    tower_grads.append(grads)
                    tower_input_loss.append(loss)
                    sn_op.append(train_op)
                    X.append(X_holder)
                    y.append(y_holder)
                    p.append(predict_y)
    X = tf.concat(X, axis=0)
    y = tf.concat(y, axis=0)
    p = tf.concat(p, axis=0)
    grads = modules.average_gradients(tower_grads)
    input_loss = tf.reduce_mean(tower_input_loss)

    train_op = opt.apply_gradients(grads)
    train_op = tf.group(train_op, [tf.assign_add(global_step, 1)])
    train_op = tf.group(train_op, sn_op)
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    for i in range(500):
        images, labels = mnist.train.next_batch(config.batch_size*len(config.GPU.split(",")))
        session.run(train_op, feed_dict={X:images, y:labels})
        if i % 25 == 0:
            correct_prediction = tf.equal(tf.argmax(p, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            accuracy_value = session.run(accuracy, feed_dict={X_holder:mnist.test.images, y_holder:mnist.test.labels})
            print('step:%d accuracy:%.4f' %(i, accuracy_value))
def main():
    t0 = time.time()
    test_singleGPU()
    t1 = time.time()
    print('single_gpu:',t1-t0)
    t0 = time.time()
    #test_multiGPU()
    t1 = time.time()
    print('multi_gpu:', t1 - t0)
if __name__=='__main__':
    main()

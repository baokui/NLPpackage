import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import model_mnist_simple,model_mnist_simple_mutiGPU
import modules
import config as Config
import time
def test_singleGPU():
    config = Config.Config_mnist()
    mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)
    config.nb_examples = mnist.train.num_examples
    tf.reset_default_graph()
    X_holder, y_holder, predict_y, loss, optimizer, train_op, _, accuracy = model_mnist_simple(config)
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)

    for i in range(10000):
        images, labels = mnist.train.next_batch(config.batch_size)
        session.run(train_op, feed_dict={X_holder:images, y_holder:labels})
        if i % 1000 == 0:
            accuracy_value = session.run(accuracy, feed_dict={X_holder:mnist.test.images, y_holder:mnist.test.labels})
            print('step:%d accuracy:%.4f' %(i, accuracy_value))
def test_multiGPU():
    config = Config.Config_mnist()
    mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)
    config.nb_examples = mnist.train.num_examples
    tf.reset_default_graph()
    X_holder, y_holder, predict_y, loss, optimizer, train_op, _, accuracy = model_mnist_simple_mutiGPU(config)
    session = tf.Session()
    init = tf.global_variables_initializer()
    session.run(init)
    for i in range(10000):
        images, labels = mnist.train.next_batch(config.batch_size*len(config.GPU.split(',')))
        session.run(train_op, feed_dict={X_holder: images, y_holder: labels})
        if i % 1000 == 0:
            accuracy_value = session.run(accuracy, feed_dict={X_holder: mnist.test.images, y_holder: mnist.test.labels})
            print('step:%d accuracy:%.4f' % (i, accuracy_value))
def main():
    t0 = time.time()
    test_singleGPU()
    t1 = time.time()
    print('single_gpu:',t1-t0)
    t0 = time.time()
    test_multiGPU()
    t1 = time.time()
    print('multi_gpu:', t1 - t0)
if __name__=='__main__':
    main()

import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import model_mnist_simple
import config
config = config.Config_mnist()
mnist = input_data.read_data_sets('data/MNIST_data', one_hot=True)
config.nb_examples = mnist.train.num_examples
X_holder, y_holder, predict_y, loss, optimizer, train_op = model_mnist_simple(config)
Weights = tf.Variable(tf.zeros([784, 10]))
biases = tf.Variable(tf.zeros([1,10]))
predict_y = tf.nn.softmax(tf.matmul(X_holder, Weights) + biases)
loss = tf.reduce_mean(-tf.reduce_sum(y_holder * tf.log(predict_y), 1))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train_op = optimizer.minimize(loss)

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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

def test_full_connected():
    mnist = input_data.read_data_sets('data/mnist/input_data/',one_hot=True)

    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32,shape=[None,784])
        y_true = tf.placeholder(tf.int32,shape=[None,10])

    with tf.variable_scope('fc_model'):
        weight = tf.Variable(tf.random_normal([784,10],mean=0.0,stddev=1.0),name='w')
        bias = tf.Variable(tf.random_normal(shape=[10],mean=0.0,stddev=1.0),name='b')
        y_predict = tf.matmul(x,weight) + bias

    with tf.variable_scope('soft_cross'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))

    with tf.variable_scope('optimizer'):
        train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    with tf.variable_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_predict,1),tf.argmax(y_true,1)),tf.float32))

    init_variable = tf.global_variables_initializer()

    saver = tf.train.Saver()

    tf.summary.scalar('loss',loss)
    tf.summary.scalar('accuracy',accuracy)
    tf.summary.histogram('weight',weight)
    tf.summary.histogram('bias',bias)
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(init_variable)
        file_writer = tf.summary.FileWriter('tmp/summary/test/',graph=sess.graph)

        for i in range(3000):
            mnist_x,mnist_y = mnist.train.next_batch(batch_size=50)
            sess.run(train_op,feed_dict={x:mnist_x,y_true:mnist_y})
            print('第%d步,准确率为%f'%(i,sess.run(accuracy,feed_dict={x:mnist_x,y_true:mnist_y})))

            summary = sess.run(merged,feed_dict={x:mnist_x,y_true:mnist_y})
            file_writer.add_summary(summary,i)

        saver.save(sess,'tmp/ckpt/fc_model')

    return None

def pre_full_connected():
    mnist = input_data.read_data_sets('data/mnist/input_data/',one_hot=True)

    with tf.variable_scope('data'):
        x = tf.placeholder(tf.float32,shape=[None,784])
        y_true = tf.placeholder(tf.int32,shape=[None,10])

    with tf.variable_scope('fc_model'):
        weight = tf.Variable(tf.random_normal([784,10],mean=0.0,stddev=1.0),name='w')
        bias = tf.Variable(tf.random_normal(shape=[10],mean=0.0,stddev=1.0),name='b')
        y_predict = tf.matmul(x,weight) + bias

    init_variable = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_variable)

        saver.restore(sess,'tmp/ckpt/fc_model')

        for i in range(50):
            x_test,y_test = mnist.test.next_batch(1)
            y_pre = sess.run(y_predict,feed_dict={x:x_test,y_true:y_test})
            print('第%d次预测,预测值为%d,真实值为%d'%(
                i,
                tf.argmax(y_pre,1).eval(),
                tf.argmax(y_test,1).eval()))

    return None

if __name__ == '__main__':
    # test_full_connected()
    pre_full_connected()
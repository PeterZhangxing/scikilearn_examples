import numpy as np
import pandas as pd
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

def test_graph():
    graph = tf.get_default_graph()
    print(graph)
    g = tf.Graph()
    print(g)
    with g.as_default():
        c = tf.constant(11.0)
        print(c.graph)

def test_session():
    plt = tf.placeholder(tf.float32,[None,3])
    a = tf.constant(2.0)
    b = tf.constant(3.0)
    res = tf.add(a,b)
    with tf.Session() as sess:
        print(sess.run(res))
        print(res.eval())

        print(plt.shape)
        feed_dict = {plt: [[1, 2, 3], [4, 5, 36], [2, 3, 4]]}
        print(sess.run(plt, feed_dict=feed_dict))
        print(plt.shape)

def change_shape():
    plt = tf.placeholder(tf.float32, [None, 3])
    print(plt)
    # print(plt.shape)
    plt.set_shape([2,3])
    print(plt)
    plt_reshaped = tf.reshape(plt,[3,2])
    print(plt_reshaped)
    with tf.Session() as sess:
        pass

def test_var():
    a = tf.constant(3.0, name="a")
    b = tf.constant(4.0, name="b")
    c = tf.add(a, b, name="add")
    var = tf.Variable(tf.random_normal([3,4],mean=0.0,stddev=1.0),name='variable')
    print(a)
    print(var)
    '''
    Tensor("a:0", shape=(), dtype=float32)
    Tensor("variable/read:0", shape=(3, 4), dtype=float32)
    '''
    var64 = tf.cast(var, tf.float64)
    print(var64) # Tensor("Cast:0", shape=(3, 4), dtype=float64)

    # 必须做一步显示的初始化op
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        # 把程序的图结构写入事件文件, graph:把指定的图写进事件文件当中
        # tensorboard --logdir=tmp/summary/test/
        filewriter = tf.summary.FileWriter("tmp/summary/test/", graph=sess.graph)

        print(sess.run([c,var,var64]))
        '''
        [7.0, array([[ 0.599226  , -0.60048056,  0.9046033 ,  2.307269  ],
           [ 1.6669829 , -1.3628939 ,  0.6571412 , -1.948221  ],
           [-1.0460758 ,  0.24262674,  1.4830804 , -0.8353118 ]],
          dtype=float32)]
        '''
        # print(var64.eval())

if __name__ == '__main__':
    # test_graph()
    # test_session()
    # change_shape()
    test_var()
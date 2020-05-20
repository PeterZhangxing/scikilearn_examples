import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

class MyConvNet(object):

    def __init__(self):
        self.mnist = input_data.read_data_sets('data/mnist/input_data/', one_hot=True)

    def weight_variables(self,shape):
        w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
        return w

    def bias_variables(self,shape):
        b = tf.Variable(tf.random_normal(shape=shape,mean=0.0,stddev=1.0))
        return b

    def build_model(self):
        # 1 prepare train data placeholder
        with tf.variable_scope('data'):
            x = tf.placeholder(tf.float32,[None,784])
            y_true = tf.placeholder(tf.int32,[None,10])

        with tf.variable_scope('conv1'):
            w_conv1 = self.weight_variables([5,5,1,32])
            b_conv1 = self.bias_variables([32])
            x_reshaped = tf.reshape(x,[-1,28,28,1])
            # [-1,28,28,32]
            x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshaped,w_conv1,strides=[1,1,1,1],padding='SAME') + b_conv1)
            # [-1,14,14,32]
            x_pool1 = tf.nn.max_pool(x_relu1,[1,2,2,1],[1,2,2,1],'SAME')

        with tf.variable_scope('conv2'):
            w_conv2 = self.weight_variables([5,5,32,64])
            b_conv2 = self.bias_variables([64])
            # [-1,14,14,64]
            x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1,w_conv2,[1,1,1,1],'SAME') + b_conv2)
            # [-1,7,7,64]
            x_pool2 = tf.nn.max_pool(x_relu2,[1,2,2,1],[1,2,2,1],"SAME")

        with tf.variable_scope('fconn'):
            reshaped_pool = tf.reshape(x_pool2,[-1,7*7*64])
            w_fconn = self.weight_variables([7*7*64,10])
            b_fconn = self.bias_variables([10])
            y_predict = tf.matmul(reshaped_pool,w_fconn) + b_fconn

        return x, y_true, y_predict

    def conv_fc_train(self):
        x, y_true, y_predict = self.build_model()
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_predict))
        train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(y_true,1),tf.arg_max(y_predict,1)),tf.float32))

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)

            if os.path.exists('tmp/ckpt/checkpoint'):
                saver.restore(sess,'tmp/ckpt/cnn_model')

            for i in range(6000):
                mnist_x, mnist_y = self.mnist.train.next_batch(50)
                sess.run(train_op, feed_dict={x: mnist_x, y_true: mnist_y})
                acc_val = sess.run(accuracy,feed_dict={x: mnist_x, y_true: mnist_y})
                print("训练第%d步,准确率为:%f" % (i,acc_val))

            saver.save(sess,'tmp/ckpt/cnn_model')

        return None

    def conv_fc_test(self):
        x, y_true, y_predict = self.build_model()

        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init_op)
            saver.restore(sess,'tmp/ckpt/cnn_model')

            for i in range(50):
                x_test,y_test = self.mnist.test.next_batch(1)
                y_pre = sess.run(y_predict,feed_dict={x:x_test,y_true:y_test})
                print('第%d次预测,预测值为%d,真实值为%d' % (
                    i,
                    tf.argmax(y_pre, 1).eval(),
                    tf.argmax(y_test, 1).eval()))

        return None

if __name__ == '__main__':
    mcn = MyConvNet()
    # mcn.conv_fc_train()
    mcn.conv_fc_test()
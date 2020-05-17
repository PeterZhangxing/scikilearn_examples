import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.app.flags.DEFINE_integer("max_step",1000,"模型训练的步数")
tf.app.flags.DEFINE_string("model_dir",'tmp/ckpt/model',"模型文件的加载的路径")

FLAGS = tf.app.flags.FLAGS

def myregression():
    with tf.variable_scope("target_data"):
        # 1、准备数据，x 特征值 [100, 1]   y 目标值[100]
        x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")
        y_ture = tf.matmul(x,[[0.7]]) + 0.8

    with tf.variable_scope("model"):
        # 2、建立线性回归模型 1个特征，1个权重， 一个偏置 y = x w + b
        # 随机给一个权重和偏置的值，让他去计算损失，然后再当前状态下优化
        # 用变量定义才能优化
        # trainable参数：指定这个变量能跟着梯度下降一起优化
        weight = tf.Variable(tf.random_normal([1,1],mean=0.0,stddev=1.0),name='weight')
        bias = tf.Variable(initial_value=0.0,name='bias')
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope('cal_loss'):
        loss = tf.reduce_mean(tf.square(y_ture-y_predict))

    with tf.variable_scope('optimizer'):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weights", weight)
    tf.summary.scalar("scalar", bias)
    # 定义合并tensor的op
    merged = tf.summary.merge_all()

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 定义一个保存模型的实例
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        print("随机初始化的参数权重为：%f, 偏置为：%f" % (weight.eval(), bias.eval()))

        filewriter = tf.summary.FileWriter("tmp/summary/test/", graph=sess.graph)
        if os.path.exists('tmp/ckpt/checkpoint'):
            saver.restore(sess,'tmp/ckpt/model')

        for i in range(1000):
            sess.run(train_op)
            summary = sess.run(merged)
            filewriter.add_summary(summary,i)
            print("第%d次优化的参数权重为：%f, 偏置为：%f" % (i, weight.eval(), bias.eval()))

        saver.save(sess,'tmp/ckpt/model')

    return None

def myregression_flags():
    with tf.variable_scope("target_data"):
        # 1、准备数据，x 特征值 [100, 7]   y 目标值[100]
        x = tf.random_normal([100, 7], mean=1.75, stddev=0.5, name="x_data")
        # true_weights = [[0.7],[0.6],[0.5],[0.4],[0.3],[0.2],[0.1]]
        # true_bias = 0.8
        y_ture = tf.matmul(x,[[0.7],[0.6],[0.5],[0.4],[0.3],[0.2],[0.1]]) + 0.8

    with tf.variable_scope("model"):
        # 2、建立线性回归模型 1个特征，1个权重， 一个偏置 y = x w + b
        # 随机给一个权重和偏置的值，让他去计算损失，然后再当前状态下优化
        # 用变量定义才能优化
        # trainable参数：指定这个变量能跟着梯度下降一起优化
        weight = tf.Variable(tf.random_normal([7,1],mean=0.0,stddev=1.0),name='weight')
        bias = tf.Variable(initial_value=0.0,name='bias')
        y_predict = tf.matmul(x, weight) + bias

    with tf.variable_scope('cal_loss'):
        loss = tf.reduce_mean(tf.square(y_ture-y_predict))

    with tf.variable_scope('optimizer'):
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    tf.summary.scalar("losses", loss)
    tf.summary.histogram("weights", weight)
    tf.summary.scalar("scalar", bias)
    # 定义合并tensor的op
    merged = tf.summary.merge_all()

    # 定义一个初始化变量的op
    init_op = tf.global_variables_initializer()

    # 定义一个保存模型的实例
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        print("随机初始化的参数权重为：",weight.eval(),end='   ')
        print("随机初始化的参数偏置为：",bias.eval())

        filewriter = tf.summary.FileWriter("tmp/summary/test/", graph=sess.graph)
        if os.path.exists(os.path.join(FLAGS.model_dir.rsplit(os.sep,1)[0],'checkpoint')):
            saver.restore(sess,FLAGS.model_dir)

        for i in range(FLAGS.max_step):
            sess.run(train_op)
            summary = sess.run(merged)
            filewriter.add_summary(summary,i)

            # print("第%d次优化的参数权重为："%i, weight.eval(), end='   ')
            # print("第%d次优化的参数偏置为："%i, bias.eval())

        # saver.save(sess,'tmp/ckpt/model')
        saver.save(sess,FLAGS.model_dir)

        print("权重为：", weight.eval(), end='   ')
        print("偏置为：", bias.eval())
    return None

if __name__ == '__main__':
    # myregression()
    myregression_flags()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("captcha_dir", "data/captcha.tfrecords", "验证码数据的路径")
tf.app.flags.DEFINE_integer("batch_size", 100, "每批次训练的样本数")
tf.app.flags.DEFINE_integer("label_num", 4, "每个样本的目标值数量")
tf.app.flags.DEFINE_integer("letter_num", 26, "每个目标值取的字母的可能心个数")

def read_and_decode():
    file_q = tf.train.string_input_producer([FLAGS.captcha_dir])

    reader = tf.TFRecordReader()
    key,val = reader.read(file_q)

    features = tf.parse_single_example(val,features={
        "image":tf.FixedLenFeature([],tf.string),
        "label":tf.FixedLenFeature([],tf.string)
    })

    image = tf.decode_raw(features["image"], tf.uint8)
    label = tf.decode_raw(features["label"], tf.uint8)
    image_reshaped = tf.reshape(image, [20, 80, 3])
    label_reshaped = tf.reshape(label, [4])

    image_batch, label_batch = tf.train.batch(
        [image_reshaped, label_reshaped],
        batch_size=FLAGS.batch_size,
        num_threads=1,
        capacity=FLAGS.batch_size
    )

    return image_batch, label_batch

def weight_variables(shape):
    w = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return w

def bias_variables(shape):
    b = tf.Variable(tf.random_normal(shape=shape, mean=0.0, stddev=1.0))
    return b

def fc_model(image_batch):
    with tf.variable_scope("model"):
        image_reshaped = tf.reshape(image_batch,[-1,20*80*3])
        weights = weight_variables([20*80*3,4*26])
        bias = bias_variables([4*26])
        y_pre = tf.matmul(tf.cast(image_reshaped, tf.float32), weights) + bias

    return y_pre

def cnn_model(image_batch):
    with tf.variable_scope('conv1'):
        w_conv1 = weight_variables([3, 3, 3, 32])
        b_conv1 = bias_variables([32])
        x_reshaped = tf.cast(image_batch,tf.float32)
        # [-1,20, 80, 3]
        x_relu1 = tf.nn.relu(tf.nn.conv2d(x_reshaped, w_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)
        # [-1,10,40,32]
        x_pool1 = tf.nn.max_pool(x_relu1, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')
        x_pool1 = tf.nn.dropout(x_pool1,0.75)

    with tf.variable_scope('conv2'):
        w_conv2 = weight_variables([3, 3, 32, 64])
        b_conv2 = bias_variables([64])
        # [-1,10,40,64]
        x_relu2 = tf.nn.relu(tf.nn.conv2d(x_pool1, w_conv2, [1, 1, 1, 1], 'SAME') + b_conv2)
        # [-1,5,20,64]
        x_pool2 = tf.nn.max_pool(x_relu2, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        x_pool2 = tf.nn.dropout(x_pool2, 0.75)

    with tf.variable_scope('conv3'):
        w_conv3 = weight_variables([3, 3, 64, 64])
        b_conv3 = bias_variables([64])
        # [-1,5,20,64]
        x_relu3 = tf.nn.relu(tf.nn.conv2d(x_pool2, w_conv3, [1, 1, 1, 1], 'SAME') + b_conv3)
        # [-1,3,10,64]
        x_pool3 = tf.nn.max_pool(x_relu3, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")
        x_pool3 = tf.nn.dropout(x_pool3, 0.75)

    with tf.variable_scope("model"):
        image_reshaped = tf.reshape(x_pool3,[-1,3*10*64])
        weights = weight_variables([3*10*64,4*26])
        bias = bias_variables([4*26])
        y_pre = tf.matmul(tf.cast(image_reshaped, tf.float32), weights) + bias

    return y_pre

def lables_to_onehot(label_batch):
    y_true = tf.one_hot(label_batch,depth=FLAGS.letter_num,on_value=1.0,axis=2)
    return y_true

def captcha_recgonise():
    image_batch, label_batch = read_and_decode()

    # y_pre = fc_model(image_batch)
    y_pre = cnn_model(image_batch)

    y_true = lables_to_onehot(label_batch)

    with tf.variable_scope("softmax_mean"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=tf.reshape(y_true,[FLAGS.batch_size, FLAGS.label_num * FLAGS.letter_num]),
            logits=y_pre
        ))

    with tf.variable_scope("optimizer"):
        train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
        # train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    with tf.variable_scope("accuracy"):
        equal_list = tf.equal(tf.argmax(y_true,dimension=2),tf.argmax(tf.reshape(y_pre,[FLAGS.batch_size, FLAGS.label_num, FLAGS.letter_num]),dimension=2))
        accuracy = tf.reduce_mean(tf.cast(equal_list, tf.float32))

    init_var = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_var)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess,coord)

        for i in range(5000):
            sess.run(train_op)
            print("第%d批次的准确率为：%f" % (i, accuracy.eval()))

        coord.request_stop()
        coord.join(threads)

    return None

if __name__ == '__main__':
    captcha_recgonise()
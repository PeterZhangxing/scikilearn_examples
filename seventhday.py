import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

tf.app.flags.DEFINE_string('job_name'," ",'ps or worker')
tf.app.flags.DEFINE_integer('task_index',0,'task number')

FLAGS = tf.app.flags.FLAGS

def main(argv):
    global_step = tf.contrib.framework.get_or_create_global_step()
    cluster = tf.train.ClusterSpec({"ps":["192.168.1.128:2223"],"worker":["192.168.1.1:2223"]})
    server = tf.train.Server(cluster,job_name=FLAGS.job_name,task_index=FLAGS.task_index)

    if FLAGS.job_name == 'ps':
        server.join()
    else:
        worker_device = "/job:worker/task:0/cpu:0/"
        with tf.device(tf.train.replica_device_setter(
                worker_device=worker_device,
                cluster=cluster
        )):
            x = tf.Variable([[1,2,3,4]])
            w = tf.Variable([[1],[2],[3],[4]])
            b = tf.Variable([10])
            y = tf.matmul(x,w) + b

        with tf.train.MonitoredTrainingSession(
            master="grpc://192.168.1.1:2223",
            is_chief=(FLAGS.task_index==0),
            config=tf.ConfigProto(log_device_placement=True),
            hooks=[tf.train.StopAtStepHook(last_step=200)]
        ) as mon_sess:
            while not mon_sess.should_stop():
                print(mon_sess.run(y))

if __name__ == '__main__':
    tf.app.run()
from test._base import BaseTest
from RLA.easy_log.tester import exp_manager
from RLA.easy_log import logger
from RLA.easy_log.complex_data_recorder import MatplotlibRecorder as mpr
import os


class ManagerTest(BaseTest):
    def _init_proj(self):
        task_name = 'test_manger_demo_task'
        rla_data_root = '../../test_data_root'
        exp_manager.configure(task_name, private_config_path='../rla_config.yaml', data_root=rla_data_root)
        exp_manager.log_files_gen()
        exp_manager.print_args()

    def test_proj_init(self):
        hp = {
            'hp1': 1,
            'hp2': 2,
        }
        exp_manager.set_hyper_param(**hp)
        exp_manager.add_record_param(['hp1'])
        self._init_proj()

    def test_log_tf(self):
        kwargs = {
            'input_size': 2,
            'learning_rate': 0.0001,
        }
        exp_manager.set_hyper_param(**kwargs)
        exp_manager.add_record_param(['input_size'])
        self._init_proj()
        import tensorflow as tf
        import numpy as np
        X_ph = tf.placeholder(dtype=tf.float32, shape=[None, kwargs["input_size"]], name='x')
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
        l = X_ph
        for _ in range(3):
            l = tf.nn.tanh(tf.layers.dense(l, 64, kernel_initializer=tf.keras.initializers.glorot_normal))

        out = tf.layers.dense(l, 1, kernel_initializer=tf.keras.initializers.glorot_normal)
        loss = tf.reduce_mean(np.square(out - y_ph))
        opt = tf.train.AdamOptimizer(learning_rate=kwargs["learning_rate"]).minimize(loss)

        sess = tf.Session().__enter__()
        sess.run(tf.variables_initializer(tf.global_variables()))

        exp_manager.new_saver(var_prefix='', max_to_keep=1)

        def target_func(x):
            return np.tanh(np.mean(x, axis=-1, keepdims=True))

        for i in range(0, 1000):
            exp_manager.time_step_holder.set_time(i)
            x_input = np.random.normal(0, 3, [64, kwargs["input_size"]])
            y = target_func(x_input)
            loss_out, y_pred = sess.run([loss, out, opt], feed_dict={X_ph: x_input, y_ph: y})[:-1]
            logger.ma_record_tabular("perf/mse", loss_out, 10)
            logger.record_tabular("y_out", np.mean(y))
            logger.dump_tabular()
            if i % 100 == 0:
                exp_manager.save_checkpoint()
            if i % 10 == 0:
                def plot_func():
                    import matplotlib.pyplot as plt
                    testX = np.repeat(np.expand_dims(np.arange(-10, 10, 0.1), axis=-1), repeats=16, axis=-1)
                    testY = target_func(testX)
                    predY = sess.run(out, feed_dict={X_ph: testX})
                    plt.plot(testX.mean(axis=-1), predY.mean(axis=-1), label='pred')
                    plt.plot(testX.mean(axis=-1), testY.mean(axis=-1), label='real')
                mpr.pretty_plot_wrapper('react_func', plot_func, xlabel='x', ylabel='y', title='react test')
            logger.dump_tabular()
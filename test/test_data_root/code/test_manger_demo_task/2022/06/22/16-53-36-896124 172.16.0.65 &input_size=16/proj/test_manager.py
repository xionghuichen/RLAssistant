from test._base import BaseTest
from RLA.easy_log.tester import exp_manager
from RLA.easy_log import logger
from RLA.easy_log.complex_data_recorder import MatplotlibRecorder as mpr
import numpy as np
import os


def target_func(x):
    return np.tanh(np.mean(x, axis=-1, keepdims=True))

class ManagerTest(BaseTest):

    def _init_proj(self, config_name='rla_config.yaml'):
        task_name = 'test_manger_demo_task'
        rla_data_root = '../../test_data_root'
        exp_manager.configure(task_name, private_config_path=f'../{config_name}', data_root=rla_data_root)
        exp_manager.log_files_gen()
        exp_manager.print_args()

    def test_log_tf(self):
        kwargs = {
            'input_size': 16,
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
        # build a neural network
        for _ in range(3):
            l = tf.nn.tanh(tf.layers.dense(l, 64, kernel_initializer=tf.keras.initializers.glorot_normal))

        out = tf.layers.dense(l, 1, kernel_initializer=tf.keras.initializers.glorot_normal)
        loss = tf.reduce_mean(np.square(out - y_ph))
        opt = tf.train.AdamOptimizer(learning_rate=kwargs["learning_rate"]).minimize(loss)
        sess = tf.Session().__enter__()
        sess.run(tf.variables_initializer(tf.global_variables()))

        exp_manager.new_saver(var_prefix='', max_to_keep=1)
        # synthetic target function.

        for i in range(0, 100):
            exp_manager.time_step_holder.set_time(i)
            x_input = np.random.normal(0, 3, [64, kwargs["input_size"]])
            y = target_func(x_input)
            loss_out, y_pred = sess.run([loss, out, opt], feed_dict={X_ph: x_input, y_ph: y})[:-1]
            # moving averaged
            logger.ma_record_tabular("perf/mse", loss_out, 10)
            logger.record_tabular("y_out", np.mean(y))
            logger.dump_tabular()
            if i % 20 == 0:
                exp_manager.save_checkpoint()
            if i % 10 == 0:
                def plot_func():
                    import matplotlib.pyplot as plt
                    testX = np.repeat(np.expand_dims(np.arange(-10, 10, 0.1), axis=-1), repeats=kwargs["input_size"], axis=-1)
                    testY = target_func(testX)
                    predY = sess.run(out, feed_dict={X_ph: testX})
                    plt.plot(testX.mean(axis=-1), predY.mean(axis=-1), label='pred')
                    plt.plot(testX.mean(axis=-1), testY.mean(axis=-1), label='real')
                mpr.pretty_plot_wrapper('react_func', plot_func, xlabel='x', ylabel='y', title='react test')
            logger.dump_tabular()


    def test_load_checkpoint_tf(self):
        pass

    def test_log_torch(self):
        kwargs = {
            'input_size': 16,
            'learning_rate': 0.0001,
        }
        exp_manager.set_hyper_param(**kwargs)
        exp_manager.add_record_param(['input_size'])
        self._init_proj(config_name='rla_config_torch.yaml')
        from torch_net import MLP, get_device
        from torch import nn
        from torch.nn import functional as F
        import torch as th
        mlp = MLP(feature_dim=kwargs['input_size'], net_arch=[64, 64, 64], activation_fn=nn.Tanh)
        exp_manager.new_saver(var_prefix='', max_to_keep=1)
        optimizer = th.optim.Adam(mlp.parameters(), lr=3e-4)
        for i in range(0, 100):
            exp_manager.time_step_holder.set_time(i)
            x_input = np.random.normal(0, 3, [64, kwargs["input_size"]])
            y = target_func(x_input)
            mse_loss = F.mse_loss(mlp(th.as_tensor(x_input).to(get_device('auto'))), y)
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            logger.ma_record_tabular("perf/mse", mse_loss.numpy(), 10)
            logger.record_tabular("y_out", np.mean(y))
            pass
            logger.dump_tabular()

    def test_load_checkpoint_torch(self):
        pass
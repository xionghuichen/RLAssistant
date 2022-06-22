from test._base import BaseTest
from RLA.easy_log.tester import exp_manager
from RLA.easy_log import logger
from RLA.easy_log.complex_data_recorder import MatplotlibRecorder as mpr
import numpy as np
from RLA.utils.utils import load_yaml
import os


def target_func(x):
    return np.tanh(np.mean(x, axis=-1, keepdims=True))


RLA_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATABASE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CODE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class ManagerTest(BaseTest):

    def _load_rla_config(self):
        return load_yaml(os.path.join(RLA_REPO_ROOT, 'example/rla_config.yaml'))

    def _init_proj(self, config_yaml, **kwargs):
        task_name = 'test_manger_demo_task'
        rla_data_root = os.path.join(DATABASE_ROOT, 'test_data_root')
        config_yaml['BACKUP_CONFIG']['backup_code_dir'] = ['proj']
        exp_manager.configure(task_name, private_config_path=config_yaml, data_root=rla_data_root,
                              code_root=CODE_ROOT, **kwargs)
        exp_manager.log_files_gen()
        exp_manager.print_args()

    def test_log_tf(self):
        kwargs = {
            'input_size': 16,
            'learning_rate': 0.0001,
        }
        exp_manager.set_hyper_param(**kwargs)
        exp_manager.add_record_param(['input_size'])
        yaml = self._load_rla_config()
        self._init_proj(yaml)
        import tensorflow as tf
        import numpy as np
        X_ph = tf.placeholder(dtype=tf.float32, shape=[None, kwargs["input_size"]], name='x')
        y_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
        l = X_ph
        # build a neural network
        for _ in range(3):
            l = tf.nn.tanh(tf.layers.dense(l, 64, kernel_initializer=tf.keras.initializers.glorot_normal))

        out = tf.layers.dense(l, 1, kernel_initializer=tf.keras.initializers.glorot_normal)
        loss = tf.reduce_mean(tf.square(out - y_ph))
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
        yaml = self._load_rla_config()
        yaml['DL_FRAMEWORK'] = 'torch'
        self._init_proj(yaml)
        from test.test_proj.proj.torch_net import MLP, to_tensor
        from torch import nn
        from torch.nn import functional as F
        import torch as th
        mlp = MLP(feature_dim=kwargs['input_size'], net_arch=[64, 64, 64], activation_fn=nn.Tanh)
        exp_manager.new_saver(var_prefix='', max_to_keep=1)
        optimizer = th.optim.Adam(mlp.parameters(), lr=kwargs['learning_rate'])
        for i in range(0, 100):
            exp_manager.time_step_holder.set_time(i)
            x_input = np.random.normal(0, 3, [64, kwargs["input_size"]])
            x_input = x_input.astype(np.float32)
            y = target_func(x_input)
            mse_loss = F.mse_loss(mlp(to_tensor(x_input)), to_tensor(y))
            optimizer.zero_grad()
            mse_loss.backward()
            optimizer.step()
            logger.ma_record_tabular("perf/mse", np.mean(mse_loss.detach().cpu().numpy()), 10)
            logger.record_tabular("y_out", np.mean(y))
            if i % 10 == 0:
                def plot_func():
                    import matplotlib.pyplot as plt
                    testX = np.repeat(np.expand_dims(np.arange(-10, 10, 0.1), axis=-1), repeats=kwargs["input_size"], axis=-1)
                    testX = testX.astype(np.float32)
                    testY = target_func(testX)
                    predY = mlp(to_tensor(testX)).detach().cpu().numpy()
                    plt.plot(testX.mean(axis=-1), predY.mean(axis=-1), label='pred')
                    plt.plot(testX.mean(axis=-1), testY.mean(axis=-1), label='real')
                mpr.pretty_plot_wrapper('react_func', plot_func, xlabel='x', ylabel='y', title='react test')
            if i % 20 == 0:
                exp_manager.save_checkpoint(model_dict={'mlp': mlp.state_dict(), 'opt': optimizer.state_dict(), 'epoch': i})
                pass
            logger.dump_tabular()

    def test_load_checkpoint_torch(self):
        pass

    def test_sent_to_master(self):
        kwargs = {
            'input_size': 16,
            'learning_rate': 0.0001,
        }
        exp_manager.set_hyper_param(**kwargs)
        exp_manager.add_record_param(['input_size'])
        yaml = self._load_rla_config()
        from test.test_proj.proj import private_config
        yaml['DL_FRAMEWORK'] = 'torch'
        yaml['SEND_LOG_FILE'] = True
        yaml['REMOTE_SETTING']['ftp_server'] = '127.0.0.1'
        yaml['REMOTE_SETTING']['file_transfer_protocol'] = 'sftp'
        yaml['REMOTE_SETTING']['username'] = private_config.username
        yaml['REMOTE_SETTING']['password'] = private_config.password
        yaml['REMOTE_SETTING']['remote_log_root'] = private_config.remote_root

        self._init_proj(yaml, is_master_node=False)
        for i in range(0, 100):
            exp_manager.time_step_holder.set_time(i)
            logger.record_tabular("i", i)
            logger.dump_tabular()
            if i % 10 == 0:
                exp_manager.sync_log_file()

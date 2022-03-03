import gym

from ppo import PPO
import argparse
from RLA.rla_argparser import arg_parser_postprocess
from RLA.easy_log.tester import exp_manager
from RLA.easy_log import logger

def mujoco_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
    parser.add_argument('--policy_type',  type=str, default='MlpPolicy')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--total_timesteps', type=int, default=int(1e5))
    # [RLA] add RLA parameters
    parser = arg_parser_postprocess(parser)
    return parser


args = mujoco_arg_parser().parse_args()
# [RLA] config RLA.
task_name = 'demo_task'
exp_manager.set_hyper_param(**vars(args))
exp_manager.add_record_param(["info", "seed", 'env'])
exp_manager.configure(task_name, private_config_path='../rla_config.yaml', data_root='../')
exp_manager.log_files_gen()
exp_manager.print_args()


env = gym.make(args.env)

model = PPO(args.policy_type, env, verbose=1, seed=args.seed)
# [RLA] mask the function name of logger to be consistent with the one in sb3.
logger.record = logger.record_tabular
logger.dump = logger.dump_tabular
model._logger = logger
model._custom_logger = True

model.learn(total_timesteps=args.total_timesteps)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()

env.close()
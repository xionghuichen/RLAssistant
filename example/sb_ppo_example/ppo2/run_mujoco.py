#!/usr/bin/env python3
import numpy as np
import gym

# from stable_baselines.common.cmd_util import mujoco_arg_parser
from stable_baselines import bench
from stable_baselines.common import set_global_seeds
from stable_baselines.common.vec_env.vec_normalize import VecNormalize
from ppo2 import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines.common.cmd_util import arg_parser
from RLA.rla_argparser import arg_parser_postprocess
from RLA.easy_log.tester import exp_manager
from RLA.easy_log import logger

def train(env_id, num_timesteps, seed):
    """
    Train PPO2 model for Mujoco environment, for testing purposes

    :param env_id: (str) the environment id string
    :param num_timesteps: (int) the number of timesteps to run
    :param seed: (int) Used to seed the random generator.
    """
    def make_env():
        env_out = gym.make(env_id)
        # env_out = bench.Monitor(env_out, logger.get_dir(), allow_early_resets=True)
        return env_out

    env = DummyVecEnv([make_env])
    env = VecNormalize(env)

    set_global_seeds(seed)
    policy = MlpPolicy
    model = PPO2(policy=policy, env=env, n_steps=2048, nminibatches=32, lam=0.95, gamma=0.99, noptepochs=10,
                 ent_coef=0.0, learning_rate=3e-4, cliprange=0.2, verbose=True)
    model.learn(total_timesteps=num_timesteps)

    return model, env


def mujoco_arg_parser():
    """
    Create an argparse.ArgumentParser for run_mujoco.py.

    :return:  (ArgumentParser) parser {'--env': 'Reacher-v2', '--seed': 0, '--num-timesteps': int(1e6), '--play': False}
    """
    parser = arg_parser()
    parser.add_argument('--env', help='environment ID', type=str, default='Reacher-v2')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num-timesteps', type=int, default=int(1e6))
    parser.add_argument('--play', default=False, action='store_true')
    # [RLA] phase 1: add RLA parameters
    parser = arg_parser_postprocess(parser)
    return parser


def main():
    """
    Runs the test
    """
    args = mujoco_arg_parser().parse_args()
    # [RLA] phase 2: config RLA.
    task_name = 'demo_task'
    exp_manager.set_hyper_param(**vars(args))
    exp_manager.add_record_param(["info", "seed", 'env'])
    exp_manager.configure(task_name, private_config_path='../rla_config.yaml', data_root='../')
    exp_manager.log_files_gen()
    exp_manager.print_args()
    # [RLA] optional:
    model, env = train(args.env, num_timesteps=args.num_timesteps, seed=args.seed)

    if args.play:
        logger.log("Running trained model")
        obs = np.zeros((env.num_envs,) + env.observation_space.shape)
        obs[:] = env.reset()
        while True:
            actions = model.step(obs)[0]
            obs[:] = env.step(actions)[0]
            env.render()


if __name__ == '__main__':
    main()

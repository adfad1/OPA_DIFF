import tensorflow as tf
from path import *
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines.ppo1 import mlp_policy, pposgd_simple
from hyperparams import *
import importlib
import random



def train(env, num_timesteps, seed):
    #from baselines.ppo1 import pposgd_simple, cnn_policy
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    def policy_fn(name, ob_space, ac_space):
        #return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space,**cnn_config)
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            **mlp_config)
    env.seed(seed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=15,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=9,
            gamma=0.99, lam=0.95, schedule='linear'
        )

def main():
    EnvModule = importlib.import_module(env_id)
    env = getattr(EnvModule, env_id)(**env_config)

    train(env, num_timesteps=12000, seed = 0)

if __name__ == '__main__':
    main()

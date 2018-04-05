import tensorflow as tf
from baselines.common import set_global_seeds, tf_util as U
from baselines import bench
from baselines.ppo1 import mlp_policy, pposgd_simple
from baselines.Environments.OPA import OPA
#from bsl.BaseEnvironment import BaseEnv
#from OPA_m import OPA_m
#from sram.sram import SRAM_Enviornment
#from OPA_Environment import *
import random

def train(env, num_timesteps, seed):
    #from baselines.ppo1 import pposgd_simple, cnn_policy
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    set_global_seeds(seed)
    def policy_fn(name, ob_space, ac_space):
        #return cnn_policy.CnnPolicy(name=name, ob_space=ob_space, ac_space=ac_space)
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    env.seed(seed)
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_batch=20,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=9,
            gamma=0.99, lam=0.95, schedule='linear',sess=sess,old_model=False
        )

def main():
    Env = OPA
    status_num =18
    params_num =3
    max_step = 10
    #env = SRAM_Enviornment.Env(nb_status=status_num, nb_params=params_num, max_step=max_step)

    #kwargvs = {'status_range': status_range,
    #                                     'params_range': params_range}
    #env = OPA(nb_status=status_num, nb_params=params_num ,max_step=max_step,  **kwargvs)

    #env = OPA_m(status_num, params_num, max_step)
    train(env, num_timesteps=1000, seed = 0)

if __name__ == '__main__':
    main()

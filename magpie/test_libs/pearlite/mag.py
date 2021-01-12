import gym
import os

from stable_baselines3 import mSAC
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy, evaluate_meta_policy

from gym_fixed_wing.fixed_wing import FixedWingAircraft
from pyfly_fixed_wing_visualizer.pyfly_fixed_wing_visualizer import simrecorder

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
config_path = startingDir + "/../../prepare/fixed_wing_config.json"
print(config_path)


def make_env(config_path, rank, seed=0, info_kw=None, sim_config_kw=None):
    """
    Utility function for multiprocessed env.

    :param config_path: (str) path to gym environment configuration file
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :param info_kw: ([str]) list of entries in info dictionary that the Monitor should extract
    :param sim_config_kw (dict) dictionary of key value pairs to override settings in the configuration file of PyFly
    """

    def _init():
        env = FixedWingAircraft(config_path, sim_config_kw=sim_config_kw)
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


if __name__ == "__main__":
    N_EVAL = 30  # number of scenarios to evaluate model on
    N_EPOCHS = 3  # number of training epochs
    N_TIMESTEPS = 2000  # number of Timesteps (=Gradient steps) per epoch

    model_name = "PPO"

    num_cpu = 1
    env = SubprocVecEnv([make_env(config_path, i) for i in range(num_cpu)])

    env.env_method("set_curriculum_level", 1)
    env.set_attr("training", True)

    if model_name == "mSAC":
        meta_model = mSAC(
            "MlpPolicy",
            env,
            verbose=1,
            policy_kwargs=dict(
                net_arch=[300, 300, 300], latent_dim=5, hidden_sizes=[200, 200, 200]
            ),
        )  # ,learning_rate=0.0006)
        meta_reward = []
        meta_std = []

        (
            meta_model_mean_reward_before,
            meta_model_std_reward_before,
        ) = evaluate_meta_policy(meta_model, env, n_eval_episodes=N_EVAL)
        meta_reward.append(meta_model_mean_reward_before)
        meta_std.append(meta_model_std_reward_before)
    else:
        meta_model = PPO("MlpPolicy", env, verbose=1)

    for i in range(N_EPOCHS):

        meta_model.learn(total_timesteps=N_TIMESTEPS)
        if model_name == "mSAC":
            meta_model_mean_reward, meta_model_std_reward = evaluate_meta_policy(
                meta_model, env, n_eval_episodes=N_EVAL
            )

            meta_reward.append(meta_model_mean_reward)
            meta_std.append(meta_model_std_reward)

            print("meta_reward = ", meta_reward)
            print("meta_std = ", meta_std)

        print("epoch:", i + 1)

    env.close()

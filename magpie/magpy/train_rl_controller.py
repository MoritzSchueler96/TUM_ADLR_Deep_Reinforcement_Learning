import gym
import numpy as np
import torch

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecCheckNan,
    DummyVecEnv,
    VecNormalize,
)
from stable_baselines3 import SAC
from stable_baselines3 import mSAC
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image

from gym_fixed_wing.fixed_wing import FixedWingAircraft
from pyfly_fixed_wing_visualizer.pyfly_fixed_wing_visualizer import simrecorder
import time
import os
import shutil
import matplotlib.pyplot as plt

try:
    from evaluate_controller import evaluate_model_on_set
except:
    # from gym_fixed_wing.examples.evaluate_controller import evaluate_model_on_set
    print("Fail.")

# global variables
curriculum_level = 0.25  # Initial difficulty level of environment
curriculum_cooldown = (
    25  # Minimum number of episodes between environment difficulty adjustments
)
render_interval = 600  # Time in seconds between rendering of training episodes
test_interval = None
test_set_path = None
last_test = 0
last_render = time.time()
checkpoint_save_interval = 300
last_save = time.time()
last_ep_info = None
log_interval = 50
render_check = {"files": [], "time": time.time()}
info_kw = [
    "success",
    "control_variation",
    "end_error",
    "total_error",
    "success_time_frac",
]
sim_config_kw = {}
env = None
model = None
model_folder = None
config_path = None


def save_model(model, save_folder):
    """
    Helper function to save model checkpoint.
    :param model: (PPO2 object) the model to save.
    :param save_folder: (str) the folder to save model to.
    :return:
    """
    model.save(os.path.join(save_folder, "model.pkl"))
    model.env.save(os.path.join(save_folder, "env.pkl"))


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        global curriculum_level, last_ep_info, info_kw, log_interval, curriculum_cooldown, render_interval, last_render, render_check, model_folder, test_interval, last_test, checkpoint_save_interval, last_save, test_set_path, config_path, env, model

        if "ep_info_buffer" in self.locals:
            ep_info_buf = self.locals["ep_info_buffer"]
        else:
            ep_info_buf = self.locals["self"].ep_info_buffer
        if len(ep_info_buf) > 0 and ep_info_buf[-1] != last_ep_info:
            last_ep_info = ep_info_buf[-1]

            now = time.time()

            info = {}
            for ep_info in ep_info_buf:
                for k in ep_info.keys():
                    if k in info_kw:
                        if k not in info:
                            info[k] = {}
                        if isinstance(ep_info[k], dict):
                            for state, v in ep_info[k].items():
                                if state in info[k]:
                                    info[k][state].append(v)
                                else:
                                    info[k][state] = [v]
                        else:
                            if "all" in info[k]:
                                info[k]["all"].append(ep_info[k])
                            else:
                                info[k]["all"] = [ep_info[k]]
            if self.logger is not None:
                if "success" in info:
                    for measure in info_kw:
                        for k, v in info[measure].items():
                            self.logger.record(
                                key="ep_info/{}_{}".format(measure, k),
                                value=np.nanmean(v),
                            )
            elif (
                self.locals["n_steps"] % log_interval == 0
                and self.locals["n_steps"] != 0
            ):
                for info_k, info_v in info.items():
                    print(
                        "\n{}:\n\t".format(info_k)
                        + "\n\t".join(
                            [
                                "{:<10s}{:.2f}".format(k, np.nanmean(v))
                                for k, v in info_v.items()
                            ]
                        )
                    )
            if curriculum_level < 1:
                if curriculum_cooldown <= 0:
                    if np.mean(info["success"]["all"]) > curriculum_level:
                        curriculum_level = min(np.mean(info["success"]["all"]) * 2, 1)
                        env.env_method("set_curriculum_level", curriculum_level)
                        curriculum_cooldown = 15
                else:
                    curriculum_cooldown -= 1

            if now - last_render >= render_interval:
                env.env_method(
                    "render",
                    indices=0,
                    mode="plot",
                    show=False,
                    close=True,
                    save_path=os.path.join(
                        model_folder, "render", str(self.num_timesteps)
                    ),
                )
                last_render = time.time()

            if (
                test_set_path is not None
                and self.num_timesteps - last_test >= test_interval
            ):
                last_test = self.num_timesteps
                evaluate_model_on_set(
                    test_set_path,
                    model,
                    config_path=config_path,
                    num_envs=self.training_env.num_envs,
                    writer=self.logger,
                    timestep=self.num_timesteps,
                )

            if now - render_check["time"] >= 30:
                for render_file in os.listdir(os.path.join(model_folder, "render")):
                    if render_file not in render_check["files"]:
                        render_check["files"].append(render_file)

                        img = plt.imread(
                            os.path.join(*[model_folder, "render", render_file])
                        )
                        self.logger.record(
                            "pyfly/image",
                            Image(img, "HWC"),
                            exclude=("stdout", "log", "json", "csv"),
                        )

            if now - last_save >= checkpoint_save_interval:
                save_model(self.model, model_folder)
                last_save = now

        return True


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
        env = Monitor(
            env, filename=None, allow_early_resets=True, info_keywords=info_kw
        )
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def main(
    model_name,
    num_envs,
    env_config_path=None,
    train_steps=None,
    policy=None,
    disable_curriculum=True,
    test_data_path=None,
):

    global last_render, render_check, test_interval, last_save, model_folder, config_path, test_set_path, model, env
    global info_kw, curriculum_level, sim_config_kw

    last_render = time.time()
    last_save = time.time()
    render_check = {"files": [], "time": time.time()}
    test_set_path = test_data_path

    num_cpu = int(num_envs)

    if policy is None:
        policy = "MlpPolicy"

    if disable_curriculum:
        curriculum_level = 1

    if train_steps:
        training_steps = int(train_steps)
    else:
        training_steps = int(5e6)

    # sim_config_kw.update({"recorder": simrecorder(train_steps)})

    test_interval = int(
        training_steps / 5 * 5
    )  # How often in time steps during training the model is evaluated on the test set

    model_folder = os.path.join("models", model_name)
    if os.path.exists(model_folder):
        load = True
    else:
        load = False
        if env_config_path is None:
            config_path = ""
        else:
            config_path = env_config_path
        os.makedirs(model_folder)
        os.makedirs(os.path.join(model_folder, "render"))
        shutil.copy2(
            os.path.join(config_path, "fixed_wing_config.json"),
            os.path.join(model_folder, "fixed_wing_config.json"),
        )
    config_path = os.path.join(model_folder, "fixed_wing_config.json")

    env = VecNormalize(
        SubprocVecEnv(
            [
                make_env(config_path, i, info_kw=info_kw, sim_config_kw=sim_config_kw)
                for i in range(num_cpu)
            ]
        )
    )

    env.env_method("set_curriculum_level", curriculum_level)
    env.set_attr("training", True)

    if load:
        if "mSAC" in model_name:
            model = mSAC.load(
                os.path.join(model_folder, "model.pkl"),
                env=env,
                verbose=1,
                tensorboard_log=os.path.join(model_folder, "tb"),
            )
        elif "SAC" in model_name:
            model = SAC.load(
                os.path.join(model_folder, "model.pkl"),
                env=env,
                verbose=1,
                tensorboard_log=os.path.join(model_folder, "tb"),
            )
        else:
            model = PPO.load(
                os.path.join(model_folder, "model.pkl"),
                env=env,
                verbose=1,
                tensorboard_log=os.path.join(model_folder, "tb"),
            )
    else:
        if "mSAC" in model_name:
            model = mSAC(
                policy,
                env,
                verbose=1,
                policy_kwargs=dict(
                    net_arch=[300, 300, 300], latent_dim=7, hidden_sizes=[300, 300, 300]
                ),
                tensorboard_log=os.path.join(model_folder, "tb"),
            )
        elif "SAC" in model_name:
            model = SAC(
                policy,
                env,
                verbose=1,
                tensorboard_log=os.path.join(model_folder, "tb"),
            )
        else:
            model = PPO(
                policy,
                env,
                verbose=1,
                tensorboard_log=os.path.join(model_folder, "tb"),
            )
    model.learn(
        total_timesteps=training_steps,
        log_interval=log_interval,
        callback=TensorboardCallback(),
    )
    save_model(model, model_folder)
    # env.env_method("render", mode="plot", show=True, close=False)
    # env.render(block=True)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "model_name",
        help="Path to model folder. If already exists, configurations will be loaded from this folder and training will resume from checkpoint.",
    )
    parser.add_argument("num_envs", help="Number of processes for environments.")

    parser.add_argument(
        "--env-config-path",
        required=False,
        help="Path to configuration for gym environment",
    )
    parser.add_argument(
        "--train-steps", required=False, help="Number of training time steps"
    )
    parser.add_argument(
        "--policy", required=False, help="Type of policy to use (MlpPolicy or other)"
    )
    parser.add_argument(
        "--disable-curriculum",
        dest="disable_curriculum",
        action="store_true",
        required=False,
        help="If this flag is set, curriculum (i.e. gradual increase in sampling region of initial and target conditions based on proficiency) is disabled.",
    )
    parser.add_argument(
        "--test-set-path",
        required=False,
        help="Path to test set. If supplied, the model is evaluated on this test set 4 times during training.",
    )

    args = parser.parse_args()

    main(
        model_name=args.model_name,
        num_envs=args.num_envs,
        env_config_path=args.env_config_path,
        train_steps=args.train_steps,
        policy=args.policy,
        disable_curriculum=args.disable_curriculum,
        test_data_path=args.test_set_path,
    )

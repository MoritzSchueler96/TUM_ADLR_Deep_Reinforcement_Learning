import gym
import sys

import os
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import mSAC
from stable_baselines3.common.evaluation import evaluate_policy, evaluate_meta_policy

from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecCheckNan,
    DummyVecEnv,
    VecNormalize,
)

import random
import datetime
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor

import numpy as np
import torch as th
from gym import spaces

##pyfly stuff
from pyfly.pyfly import PyFly
from pyfly.pid_controller import PIDController
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from pyfly_fixed_wing_visualizer.pyfly_fixed_wing_visualizer import simrecorder

########################################################################################################################################################
# NOTE: fly to next point, no wind
#       next: fly to next point (10 mtrs) + Wind
#       next: fly to 2 pts
#       reward: delta distanz zum zielpunkt = geschwindigkeit
#       reward: differenz wischen aktuellem heading und goal heading
#       timer: szenario ende von intern triggern, i.e. zu weit von pfad weg
#       curriculum implementieren ( XX epochs no wind, reset replay buffers (keep weights--> reset buffer fkt), XX epochs some wind , .... )

import json
import copy
from collections import deque

startingDir = os.path.dirname(os.path.realpath(__file__))
print(startingDir)
os.chdir(startingDir)
configDir = "../libs/pyfly/pyfly/"
modelDir = "models/"

########################################################################################################################################################

import time
from stable_baselines3.common.logger import Image
from stable_baselines3.common.callbacks import BaseCallback


# global variables
render_interval = 50000  # Time in seconds between rendering of training episodes
test_interval = 500000
last_test = 0
last_render = time.time()
checkpoint_save_interval = 300
last_save = time.time()
last_ep_info = None
log_interval = 5
render_check = {"files": [], "time": time.time()}
info_kw = [
    "success",
    "control_variation",
    "end_error",
    "total_error",
    "success_time_frac",
]
info_kw = [
    "rew",
]
env = None
model = None
model_folder = None


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=1):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        global last_ep_info, info_kw, log_interval, render_interval, last_render, render_check, model_folder, test_interval, last_test, checkpoint_save_interval, last_save, env, model

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

            # nur alle x Zeitschritte rendern
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

            if self.num_timesteps - last_test >= test_interval:
                last_test = self.num_timesteps
                evaluate_meta_policy(
                    model,
                    env,
                    # writer=self.logger,
                )

            if now - render_check["time"] >= 300000:
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


########################################################################################################################################################


class FixedWingAircraft_simple(gym.Env):
    def __init__(
        self,
        config_path,
        task={},
        n_tasks=2,
        sampler=None,
        sim_config_path=None,
        sim_parameter_path=None,
        config_kw=None,
        sim_config_kw=None,
    ):
        """
        A gym environment for fixed-wing aircraft, interfacing the python flight simulator PyFly to the openAI environment.
        :param config_path: (string) path to json configuration file for gym environment
        :param sim_config_path: (string) path to json configuration file for PyFly
        :param sim_parameter_path: (string) path to aircraft parameter file used by PyFly
        """

        # choose simulator
        self.simulator = PyFly(
            os.path.join(configDir, "pyfly_config.json"),
            os.path.join(configDir, "x8_param.mat"),
        )

        # self.simulator.reset(state={"roll": -0.5, "pitch": 0.15, "Wind": 1})
        # self.simulator.turbulence = True
        # self.simulator.turbulence_intensity = "light" #, "moderate", "severe".

        # init sim environment
        self.history = None
        self.steps_count = None
        self._steps_for_current_target = None
        self.goal_achieved = False

        # init observation vector

        # iterate over observation states

        self.rwd_vec = ["position_n", "position_e", "position_d", "Va"]
        self.rwd_weights = [1, 1, 1, 1]
        self.rwd_method = ["euclid", "euclid", "euclid", "abs"]
        self.targets = [200, 0, -50, 22]

        self.rwd_delta = [5]

        self.steps_max = 500

        self.obs_vec = [
            "roll",
            "pitch",
            "Va",
            "omega_p",
            "omega_q",
            "omega_r",
            "elevon_left",
            "elevon_right",
            "throttle",
            "position_n",
            "position_e",
            "position_d",
        ]

        obs_low = np.ones(len(self.obs_vec)) * -np.inf
        obs_high = np.ones(len(self.obs_vec)) * np.inf

        # set observation space
        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32
        )

        # set action space
        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32,
        )

        self._task = task

        # choose starting task
        self.cur_pos = 0
        self.idx = 0

        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0].get("roll", 0.0)
        self._goal = self._goal_vel
        self.all_goals = [-1]

        # skip rendering
        self.skip = False

    def sample_tasks(num_tasks):
        tasks = []
        taskDir = os.path.join(startingDir, "tasks")
        all_files = os.listdir(taskDir)
        files = random.sample(all_files, num_tasks)
        for f in files:
            task = np.load(os.path.join(taskDir, f), allow_pickle=True)
            tasks.append(task)

        return tasks

    def sample_task(id):
        if id == self.idx and self.cur_pos < len(self.tasks[id]):
            self.cur_pos += 1
        elif id == self.idx:
            self.cur_pos = 0
        else:
            self.cur_pos = 0
            self.idx = id
        sample_target(id, self.cur_pos)
        return self.tasks[id]

    def sample_target(id, pos):
        if pos == 0:
            start = self.tasks[id][pos]
            goal = self.tasks[id][pos + 1]
        else:
            start = self.tasks[id][pos - 1]
            goal = self.tasks[id][pos]
        self.simulator.reset(state=start)
        self.target = goal  # self._goal?
        self.rec = simrecorder(self.steps_max)

    def get_all_task_idx(pself):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.sample_task(idx)

        self._goal_vel = self._task["roll"]  # not needed anymore?
        self._goal = self._goal_vel  # not needed anymore?
        print(self._goal, idx)  # not needed anymore?
        self.reset()

    def seed(self, seed=None):
        """
        Seed the random number generator of the flight simulator
        :param seed: (int) seed for random state
        """
        # used
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self.simulator.seed(seed)
        return [seed]

    def reset(self, state=None, target=None, **sim_reset_kw):
        """
        Reset state of environment.
        :param state: (dict) set initial value of states to given value.
        :param target: (dict) set initial value of targets to given value.
        :return: ([float]) observation vector
        """
        # reset steps
        self.steps_count = 0

        self.simulator = PyFly(
            os.path.join(configDir, "pyfly_config.json"),
            os.path.join(configDir, "x8_param.mat"),
        )

        # self.reset_task(id) ?
        self.simulator.reset(state={"roll": self._goal, "pitch": 0.0, "Wind": 1})
        # self.simulator.turbulence = True
        # self.simulator.turbulence_intensity = "light"

        obs = self.get_observation()

        print("--reset--")

        return obs

    def step(self, action):
        """
        Perform one step from action chosen by agent.
        :param action: ([float]) the action chosen by the agent
        :return: ([float], float, bool, dict) observation vector, reward, done, extra information about episode on done
        """
        # check if any action is nan
        assert not np.any(np.isnan(action))

        # action[2] = abs(action[2])

        if not self.skip:
            if self.simulator.cur_sim_step == 0:
                self.rec = simrecorder(self.steps_max)

        control_input = list(action)

        # simulate one step
        step_success, step_info = self.simulator.step(control_input)

        # update step count
        self.steps_count += 1
        #        self._steps_for_current_target += 1

        done = False

        # check if max step count / end of simulation reached
        if self.steps_count >= self.steps_max > 0:
            done = True

        if step_success:
            goal_achieved_on_step = False

            # save current state for visualization with pyfly fixed wing visualizer
            if not self.skip:
                if self.rec:  # and self.steps_count % 50 == 0:
                    self.rec.savestate(
                        self.simulator.state, self.simulator.cur_sim_step - 1
                    )

            # calculate reward  TODO: move down to get observation and appending?
            reward = self.get_reward(
                action=control_input,
                success=goal_achieved_on_step,
                potential=False,
            )

        else:
            # end simulation bc step failed (TODO: add reasons for failing)
            # set special reward s.t. action will not be applied again in this state
            done = True

            reward = self.steps_count - self.steps_max

        obs = self.get_observation()
        info = {"rew": reward}

        return obs, reward, done, info

    def set_skip(self, val):
        self.skip = val

    def render(
        self,
        mode="plot",
        show=True,
        close=True,
        block=False,
        save_path=None,
        epoch=None,
    ):
        """
        Visualize environment history. Plots of action and reward can be enabled through configuration file.
        :param mode: (str) render mode, one of plot for graph representation and animation for 3D animation with blender
        :param show: (bool) if true, plt.show is called, if false the figure is returned
        :param close: (bool) if figure should be closed after showing, or reused for next render call
        :param block: (bool) block argument to matplotlib blocking script from continuing until figure is closed
        :param save_path (str) if given, render is saved to this path.
        :return: (matplotlib Figure) if show is false in plot mode, the render figure is returned
        """
        if mode == "plt":
            # get target values
            targets = {
                k: {"data": np.array(v)}
                for k, v in self.history["target"].items()  # change with history
            }

            # create figure
            self.viewer = {"fig": plt.figure(figsize=(9, 16))}

            # set number of plots to create
            subfig_count = 8

            self.viewer["gs"] = matplotlib.gridspec.GridSpec(subfig_count, 1)

            # plot actions
            labels = [
                a["name"] for a in self.cfg["action"]["states"]
            ]  # change to fix names elev_left, elev_right, throttle
            x, y = (
                list(range(len(self.history["action"]))),
                np.array(
                    self.history["action"]
                ),  # change to get values for elev und throttle
            )
            ax = plt.subplot(self.viewer["gs"][-2, 0], title="Actions")
            for i in range(y.shape[1]):
                ax.plot(x, y[:, i], label=labels[i])
            ax.legend()

            # plot rewards
            x, y = (
                list(range(len(self.history["reward"]))),
                self.history["reward"],
            )  # change to get reward values
            ax = plt.subplot(self.viewer["gs"][-1, 0], title="Reward")
            ax.plot(x, y)

            # plot targets
            self.simulator.render(close=close, targets=targets, viewer=self.viewer)

            # save figures if specified
            if save_path is not None:
                if not os.path.isdir(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                _, ext = os.path.splitext(save_path)
                if ext != "":
                    plt.savefig(save_path, bbox_inches="tight", format=ext[1:])
                else:
                    plt.savefig(save_path, bbox_inches="tight")

            if show:
                plt.show(block=block)
                if close:
                    plt.close(self.viewer["fig"])
                    self.viewer = None
            else:
                if close:
                    plt.close(self.viewer["fig"])
                    self.viewer = None
                else:
                    return self.viewer["fig"]

        elif mode == "other":
            if epoch == None:
                self.rec.plot(render="other")
            else:
                self.rec.plot(render="other", epoch=epoch)

    # TODO: change to work with array
    def apply_scaling(self):
        val = self.simulator.state[r[1]].value
        max = self.rwd_max
        target = self.target.state[r[1]].value

        if target - val > target - max:
            self.rwd_max = val

        fac = (target - val) / (target - max)

        return fac

    def get_reward(self, action=None, success=False, potential=False):
        """
        Get the reward for the current state of the environment.
        :return: (float) reward
        """
        # used
        reward = 0

        # idea on how to calc reward without scaling
        rew = 0
        for r in enumerate(self.rwd_vec):
            if self.rwd_method[r[0]] == "abs":
                rew += self.rwd_weights[r[0]] * (
                    self.simulator.state[r[1]].value
                    - self.simulator.state[r[1]].history[-2]
                )
            else:
                rew += (
                    self.rwd_weights[r[0]]
                    * np.exp(
                        self.simulator.state[r[1]].value
                        - self.simulator.state[r[1]].history[-2]
                    )
                    ** 2
                )

        # for idx, meas in enumerate(self.rwd_vec):
        #     if self.targets[idx] != 0:
        #         val = -0.02* ((self.simulator.state[meas].value - self.targets[idx])/(self.targets[idx]))**2 #Always fly to 100,0,-50
        #     else:
        #         val = -0.02* (self.simulator.state[meas].value - self.targets[idx])**2 #Always fly to 100,0,-50

        #     reward += val
        curr_hgt = -1 * self.simulator.state["position_d"].value
        last_hgt = -1 * self.simulator.state["position_d"].history[-2]

        # negatives delta zum vorherigen zeitschritt
        #

        reward = curr_hgt - last_hgt
        # reward -= np.sum(np.asarray(action)**2)
        return reward

    def get_observation(self):
        """
        Get the observation vector for current state of the environment.
        :return: ([float]) observation vector
        """
        # used
        obs = []

        for obb in self.obs_vec:
            obs.append(self.simulator.state[obb].value)

        return np.array(obs)


########################################################################################################################################################


def make_env(config_path, n_tasks, rank=0, seed=0, info_kw=None, sim_config_kw=None):
    """
    Utility function for multiprocessed env.

    :param config_path: (str) path to gym environment configuration file
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :param info_kw: ([str]) list of entries in info dictionary that the Monitor should extract
    :param sim_config_kw (dict) dictionary of key value pairs to override settings in the configuration file of PyFly
    """

    def _init():
        env = FixedWingAircraft_simple(config_path, n_tasks=n_tasks)
        env = Monitor(
            env, filename=None, allow_early_resets=True, info_keywords=info_kw
        )
        env.seed(seed + rank)
        return env

    set_random_seed(seed)
    return _init


def save_model(model, save_folder):
    """
    Helper function to save model checkpoint.
    :param model: (PPO2 object) the model to save.
    :param save_folder: (str) the folder to save model to.
    :return:
    """
    model.env.save(os.path.join(save_folder, "env.pkl"))
    model.save(
        os.path.join(save_folder, "model.pkl")
    )  # FIX failes somehow only for mSAC


if __name__ == "__main__":
    modelname = "Msac__" + datetime.datetime.now().strftime("%H_%M%p__%B_%d_%Y")
    model_folder = os.path.join(modelDir, modelname)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
        os.makedirs(os.path.join(model_folder, "logs"))
    filename = "simpol.txt"
    file = os.path.join(model_folder, "logs", filename)

    env = VecNormalize(
        SubprocVecEnv(
            [
                make_env(
                    config_path=os.path.join(configDir, "pyfly_config.json"),
                    rank=n,
                    seed=1337,
                    info_kw=info_kw,
                    n_tasks=130,
                )
                for n in range(1)
            ]
        )
    )

    if False:
        meta_model = mSAC(
            "MlpPolicy",
            env,
            n_traintasks=100,
            n_evaltasks=30,
            n_epochtasks=5,
            verbose=1,
            policy_kwargs=dict(
                net_arch=[300, 300, 300], latent_dim=5, hidden_sizes=[200, 200, 200]
            ),
            tensorboard_log=os.path.join(model_folder, "tb"),
        )  # ,learning_rate=0.0006)
    else:
        meta_model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=os.path.join(model_folder, "tb"),
        )

    meta_reward = []
    meta_std = []

    print("-Start-")
    n_eval = 30

    # meta_model_mean_reward_before, meta_model_std_reward_before = evaluate_meta_policy(meta_model, env, n_eval_episodes=n_eval, epoch=0)
    meta_model_mean_reward_before = 3
    meta_model_mean_reward = 5
    meta_model_std_reward_before = 4
    meta_model_std_reward = 2
    meta_reward.append(meta_model_mean_reward_before)
    meta_std.append(meta_model_std_reward_before)

    my_file = open(file, "w+")
    with open(file, "a") as myfile:
        myfile.write("epoch:" + str(0) + "\n")
        myfile.write("meta_reward = " + str(meta_reward) + "\n")
        myfile.write("meta_std = " + str(meta_std) + "\n")
        myfile.write(
            "================================================================================================\n\n"
        )

    print(
        "##################################Start Learning##################################"
    )
    for i in range(50):

        meta_model.learn(
            total_timesteps=5 * 500,
            callback=TensorboardCallback(),  # FIX creates new tb file each time called
        )  # , eval_freq=100, n_eval_episodes=5, log_interval = 100)

        # meta_model_mean_reward, meta_model_std_reward = evaluate_meta_policy(    meta_model, env, n_eval_episodes=n_eval, epoch=i + 1)

        meta_reward.append(meta_model_mean_reward)
        meta_std.append(meta_model_std_reward)

        print("epoch:", i)
        print("meta_reward = ", meta_reward)
        print("meta_std = ", meta_std)

        with open(file, "a") as myfile:
            myfile.write("epoch:" + str(i + 1) + "\n")
            myfile.write("meta_reward = " + str(meta_reward) + "\n")
            myfile.write("meta_std = " + str(meta_std) + "\n")
            myfile.write(
                "================================================================================================\n\n"
            )

    save_model(meta_model, modelDir + modelname)
    # env.close()


########################################################################################################################################################

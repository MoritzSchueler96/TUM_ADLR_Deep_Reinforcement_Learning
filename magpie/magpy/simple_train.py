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

sys.path.append("../libs/pyfly-fixed-wing-visualizer/")
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
from stable_baselines3.common import logger # to log curriculum level
from stable_baselines3.common.logger import Image
from stable_baselines3.common.callbacks import BaseCallback


# global variables
render_interval = 500  # Time in seconds between rendering of training episodes
tb_image_send_interval = 99999999
test_interval = 500000
last_test = 0
last_render = time.time()
checkpoint_save_interval = 500000
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
    "target",
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
        global last_ep_info, info_kw, log_interval, render_interval, last_render, render_check, model_folder, test_interval, last_test, checkpoint_save_interval, last_save, env, model, tb_image_send_interval

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

            if False:  # self.num_timesteps - last_test >= test_interval:
                last_test = self.num_timesteps
                evaluate_meta_policy(
                    model,
                    env,
                    # writer=self.logger,
                )

            if now - render_check["time"] >= tb_image_send_interval:
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
        task_dir=None,
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
        # Used if trained with curriculum
        self.task_dir = task_dir

        # choose simulator
        self.simulator = PyFly(
            os.path.join(configDir, "pyfly_config.json"),
            os.path.join(configDir, "x8_param.mat"),
        )

        # self.simulator.reset(state={"roll": -0.5, "pitch": 0.15, "Wind": 1})
        # self.simulator.turbulence = True
        # self.simulator.turbulence_intensity = "light" #, "moderate", "severe".

        # init sim environment

        self.steps_count = None
        self._steps_for_current_target = None
        self.goal_achieved = False

        # init observation vector

        # iterate over observation states
        # NOTE: What about pitch, roll rewards??
        self.rwd_vec = ["position_n", "position_e", "position_d"]
        self.rwd_weights = [1, 1, 1, 1]
        self.rew_range = {"position_n": 6, "position_e": 6, "position_d": 6}
        self.target = {}

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

        self.act_vec = [
            "elevon_left",
            "elevon_right",
            "throttle",
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
        self.cur_pos = -1
        self.idx = -1

        self.tasks = self.sample_tasks(n_tasks)
        self._task = self.sample_task(0)
        self._goal = self._task[0]
        self.goal_bounds = {
            "position_n": 0.5,
            "position_e": 0.5,
            "position_d": 0.5,
            "roll": 60,
            "pitch": 25,
            "Va": 6,
        }

        # skip rendering
        self.skip = False

        # init history
        self.history = {
            "action": [],
            "reward": [],
            "observation": [],
            "target": {k: [v] for k, v in self._task[0].items()},
            "error": [],  # {k: [self._get_error(k)] for k in self.target.keys()},
            "goal": {k: [0] for k in self.rwd_vec},
        }

    def get_goal_vector(self, goal_position):
        # extract velocities and calculate Va
        Va = []
        for vel in ["velocity_u", "velocity_v", "velocity_w"]:
            Va.append(goal_position[vel])
            del goal_position[vel]

        Va = np.linalg.norm(Va)
        #goal_position["Va"] = Va

        unwanted_vars = set(goal_position.keys()) - set(self.rwd_vec)
        for unwanted_key in unwanted_vars:
            del goal_position[unwanted_key]
        return goal_position

    def sample_tasks(self, num_tasks):
        tasks = []
        if self.task_dir is not None:
            # Used in curriculum Learning
            taskDir = self.task_dir
        else:
            taskDir = os.path.join(startingDir, "tasks", "easy")
        all_files = os.listdir(taskDir)
        files = random.sample(all_files, num_tasks)
        for f in files:
            task = np.load(os.path.join(taskDir, f), allow_pickle=True)
            tasks.append(task)
        for file in files:
            print(file)
        return tasks

    def sample_task(self, id):
        if id == self.idx and self.cur_pos < len(self.tasks[id])-2:
            self.cur_pos += 1
        elif id == self.idx:
            self.cur_pos = 0
        else:
            self.cur_pos = 0
            self.idx = id
        self.sample_target(id, self.cur_pos)
        return self.tasks[id]

    def sample_target(self, id, pos):
        print(id, pos)
        start = self.tasks[id][pos].copy()
        self.simulator.reset(state=start)
        goal = self.tasks[id][pos + 1].copy()
        self._goal = self.get_goal_vector(goal)
        return self.tasks[id], goal

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        print('PEARL WANTs:', idx)
        self.idx = idx
        self.cur_pos = 0
        self._task, _ = self.sample_target(idx, 0)

        print(len(self._task))

    def seed(self, seed=None):
        """
        Seed the random number generator of the flight simulator
        :param seed: (int) seed for random state
        """
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

        self.reset_task(self.idx)  # or 0?
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

        # save action
        self.history["action"].append(action)

        # check if any action is nan
        assert not np.any(np.isnan(action))

        # action[2] = abs(action[2])

        if not self.skip:
            if self.steps_count == 0: #self.simulator.cur_sim_step == 0:

                obs = self.get_observation()
                self.history = {
                    "action": [],
                    "reward": [],
                    "observation": [obs],
                    "target": {k: [v] for k, v in self._task[0].items()},
                    "error": [],  # {k: [self._get_error(k)] for k in self.target.keys()},
                    "goal": {k: [0] for k in self.rwd_vec},
                }


                self.rec = simrecorder(self.steps_max)

        control_input = list(action)

        # simulate one step
        step_success, step_info = self.simulator.step(control_input)

        # update step count
        self.steps_count += 1
        #        self._steps_for_current_target += 1

        info = {}
        done = False

        # check if max step count / end of simulation reached
        if self.steps_count >= self.steps_max > 0:
            done = True
            info["termination"] = "steps"

        if step_success:
            goal_achieved_on_step = False
            goal_status = []

            
            for state, status in self._get_goal_status().items():
                            self.history["goal"][state].append(status)
                            goal_status.append(status)

            for state, status in self._task[self.cur_pos + 1].items():
                self.target[state] = status
                self.history["target"][state].append(status)
                # self.history["error"][state].append(self._get_error(status)) # get error?


            # save current state for visualization with pyfly fixed wing visualizer
            if not self.skip:
                if self.rec:  # and self.steps_count % 50 == 0:
                    self.rec.savestate(
                        self.simulator.state, self.steps_count - 1, self.target
                    )

            

            if all(goal_status):
                goal_achieved_on_step = True
                self.sample_task(self.idx)# + 1)
                self.goal_achieved = True
                #done = True #otherwise new sim is started

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
            info["termination"] = step_info["termination"]

        obs = self.get_observation()
        self.history["observation"].append(obs)
        self.history["reward"].append(reward)

        if done:
            # calc some metrics
            metric = 3

        info["target"] = self.target

        return obs, reward, done, info

    def _get_goal_status(self):
        """
        Get current status of whether the goal for each target state as specified by configuration is achieved.
        :return: (dict) status for each and all target states
        """
        goal_status = {}
        for state in self._goal.keys():
            err = self._get_error(state)
            goal_status[state] = np.abs(err) <= self.goal_bounds[state]

        return goal_status

    def _get_error(self, state):
        """
        Get difference between current value of state and target value.
        :param state: (string) name of state
        :return: (float) error
        """
        if state in ["pitch", "roll"]:
            return self._get_angle_dist(
                self._goal[state], self.simulator.state[state].value
            )
        else:
            return self._goal[state] - self.simulator.state[state].value

    def _get_angle_dist(self, ang1, ang2):
        """
        Get shortest distance between two angles in [-pi, pi].
        :param ang1: (float) first angle
        :param ang2: (float) second angle
        :return: (float) distance between angles
        """
        dist = (ang2 - ang1 + np.pi) % (2 * np.pi) - np.pi
        if dist < -np.pi:
            dist += 2 * np.pi

        return dist

    def save_history(self, path, states, save_targets=True):
        """
        Save environment state history to file.
        :param path: (string) path to save history to
        :param states: (string or [string]) names of states to save
        :param save_targets: (bool) save targets
        """
        self.simulator.save_history(path, states)
        if save_targets:
            res = np.load(path, allow_pickle=True).item()
            for state in self.target.keys():
                if state in res:
                    res[state + "_target"] = self.history["target"][state]
            np.save(path, res)

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
        if mode == "plot" and False:
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
            labels = self.act_vec
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
 #           self.simulator.render(close=close, targets=targets, viewer=self.viewer)

            # save figures if specified
            if save_path is not None:
                if not os.path.isdir(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                _, ext = os.path.splitext(save_path)
                if ext != "":
                    plt.savefig(save_path, format=ext[1:])
                else:
                    plt.savefig(save_path)

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

    def scale_reward(self, name, dist):
        return dist / self.rew_range[name]

    def get_reward(self, action=None, success=False, potential=False):
        """
        Get the reward for the current state of the environment.
        :return: (float) reward
        """

        # idea on how to calc reward without scaling
        rew = 0
        for num, name in enumerate(self.rwd_vec):
            dist = np.abs(self._get_error(name))
            dist = self.scale_reward(name, dist)
            rew += self.rwd_weights[num] * dist

        rew = 1 / np.exp(rew)

        return rew

    def get_observation(self):
        """
        Get the observation vector for current state of the environment.
        :return: ([float]) observation vector
        """
        obs = []

        for obb in self.obs_vec:
            obs.append(self.simulator.state[obb].value)

        return np.array(obs)


########################################################################################################################################################


def make_env(
    config_path, n_tasks, rank=0, seed=0, taskdir=None, info_kw=None, sim_config_kw=None
):
    """
    Utility function for multiprocessed env.

    :param config_path: (str) path to gym environment configuration file
    :param rank: (int) index of the subprocess
    :param seed: (int) the inital seed for RNG
    :param info_kw: ([str]) list of entries in info dictionary that the Monitor should extract
    :param sim_config_kw (dict) dictionary of key value pairs to override settings in the configuration file of PyFly
    """

    def _init():
        env = FixedWingAircraft_simple(config_path, n_tasks=n_tasks, task_dir=taskdir)
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
        os.path.join(save_folder, "model.pkl"),
        include=["actor", "critic"],
        exclude=[
            "JUST_EVAL",
            "RBList_encoder",
            "RBList_replay",
            "RBList_eval",
            "callback",
        ],
    )  # FIX failes somehow only for mSAC


if __name__ == "__main__":

    ##############
    ## Settings ##
    N_EPOCHS = 30
    N_TRAINTASKS = 50
    N_TESTTASKS = 15

    curriculum = True
    curric_idx = 0
    curric_paths = ["easy", "medium", "hard", "hard","hard"]#, "hard", "extreme", "ludicrous"]
    CURR_INC = 1
    ##############

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
                    n_tasks=(N_TESTTASKS + N_TRAINTASKS),
                )
                for n in range(1)
            ]
        )
    )

    cllbck = TensorboardCallback()

    if True:
        meta = True
        meta_model = mSAC(
            "MlpPolicy",
            env,
            n_traintasks=N_TRAINTASKS,
            n_evaltasks=N_TESTTASKS,
            n_epochtasks=5,
            verbose=1,
            policy_kwargs=dict(
                net_arch=[300, 300, 300], latent_dim=5, hidden_sizes=[200, 200, 200]
            ),
            tensorboard_log=os.path.join(model_folder, "tb"),
        )  # ,learning_rate=0.0006)

        _, cb = meta_model._setup_learn(5 * 500, eval_env=env, callback=cllbck)

        meta_model.callback = cb
    else:
        meta = False
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=os.path.join(model_folder, "tb"),
        )

    reward = []
    std = []

    

    print("-Start-")
    if meta:
        save_model(meta_model, modelDir + modelname)
        print(env)
        meta_model = mSAC.load(os.path.join(modelDir + modelname, "model.pkl"))
        meta_model.env = env
        print(env)
        print("done")
        meta_model.callback = cllbck
        model_mean_reward_before, model_std_reward_before = evaluate_meta_policy(
            meta_model, env, n_eval_episodes=N_TESTTASKS, epoch=0
        )
    else:
        model.callback = cllbck
        model_mean_reward_before, model_std_reward_before = evaluate_policy(model, env, n_eval_episodes=N_TESTTASKS)

    reward.append(model_mean_reward_before)
    std.append(model_std_reward_before)

    my_file = open(file, "w+")
    with open(file, "a") as myfile:
        myfile.write("epoch:" + str(0) + "\n")
        myfile.write("meta_reward = " + str(reward) + "\n")
        myfile.write("meta_std = " + str(std) + "\n")
        myfile.write(
            "================================================================================================\n\n"
        )

    print(
        "##################################Start Learning##################################"
    )

    for EPOCH in range(N_EPOCHS):

        logger.record(key = "train/curriculum_level", value = curric_idx)

        if meta:
            meta_model.learn(
                total_timesteps=5 * 500,
                callback=TensorboardCallback(),  # FIX creates new tb file each time called
            )  # , eval_freq=100, n_eval_episodes=5, log_interval = 100)

            model_mean_reward, model_std_reward = evaluate_meta_policy(
                meta_model, env, n_eval_episodes=N_TESTTASKS, epoch=EPOCH + 1
            )

        else:

            model.callback = cllbck

            model.learn(
                total_timesteps=5 * 500,
                callback=cllbck,#TensorboardCallback(),  # FIX creates new tb file each time called
            )  # , eval_freq=100, n_eval_episodes=5, log_interval = 100)

            model_mean_reward, model_std_reward = evaluate_policy(
                model, env, n_eval_episodes=N_TESTTASKS
            )

        reward.append(model_mean_reward)
        std.append(model_std_reward)

        print("epoch:", EPOCH)
        print("meta_reward = ", reward)
        print("meta_std = ", std)

        with open(file, "a") as myfile:
            myfile.write("epoch:" + str(EPOCH + 1) + "\n")
            myfile.write("meta_reward = " + str(reward) + "\n")
            myfile.write("meta_std = " + str(std) + "\n")
            myfile.write(
                "================================================================================================\n\n"
            )

        if curriculum and EPOCH % CURR_INC == 0 and EPOCH >= 1:
            curric_idx += 1

            if meta:

                meta_model.initial_experience = False
                meta_model.reset_buffers()

            env = VecNormalize(
                SubprocVecEnv(
                    [
                        make_env(
                            config_path=os.path.join(configDir, "pyfly_config.json"),
                            rank=0,
                            seed=1337,
                            info_kw=info_kw,
                            n_tasks=(N_TESTTASKS + N_TRAINTASKS),
                            taskdir=os.path.join(
                                startingDir, "tasks", curric_paths[curric_idx]
                            ),
                        )
                        for n in range(1)
                    ]
                )
            )
            

            if meta:
                meta_model.env = env
            else:
                model.env = env
        if meta:
            save_model(meta_model, modelDir + modelname)
        else:
            save_model(model, modelDir + modelname)
    # env.close()


########################################################################################################################################################

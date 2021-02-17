import gym
import sys

import os
sys.path.append('../libs/pyfly-fixed-wing-visualizer/')
sys.path.append('../libs/stable-baselines3/')
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import mSAC
from stable_baselines3.common.evaluation import evaluate_policy, evaluate_meta_policy

import numpy as np
import torch as th
from gym import spaces

##pyfly stuff
from pyfly.pyfly import PyFly
from pyfly.pid_controller import PIDController
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

from pyfly_fixed_wing_visualizer.pyfly_fixed_wing_visualizer import simrecorder
th.manual_seed(42)
np.random.seed(666)

########################################################################################################################################################
# NOTE: fly to next point, no wind
#       next: fly to next point (10 mtrs) + Wind
#       next: fly to 2 pts
#       reward: delta distanz zum zielpunkt = geschwindigkeit
#       reward: differenz wischen aktuellem heading und goal heading
#       timer: szenario ende von intern triggern, i.e. zu weit von pfad weg
#       curriculum implementieren ( XX epochs no wind, reset replay buffers (keep weights--> reset buffer fkt), XX epochs some wind , .... )

from pyfly.pyfly import PyFly
import json
import numpy as np
import matplotlib.pyplot as plt
import copy
from collections import deque


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
        self.simulator = PyFly("/home/user/anaconda3/lib/python3.8/site-packages/pyfly/pyfly_config.json", "/home/user/anaconda3/lib/python3.8/site-packages/pyfly/x8_param.mat")

        self.simulator.seed(0)
        #self.simulator.reset(state={"roll": -0.5, "pitch": 0.15, "Wind": 1})
        #self.simulator.turbulence = True
        #self.simulator.turbulence_intensity = "light" #, "moderate", "severe".

        # init sim environment
        self.history = None
        self.steps_count = None
        self._steps_for_current_target = None
        self.goal_achieved = False


        # init observation vector
       
        # iterate over observation states

        self.rwd_vec =  ["position_n","position_e","position_d","Va"]
        self.targets = [200, 0, -50, 22]

        self.rwd_delta = [5]

        self.steps_max = 500
            

        self.obs_vec =  ["roll",
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
                    "position_d"]

        obs_low = np.ones(len(self.obs_vec)) * -np.inf
        obs_high = np.ones(len(self.obs_vec)) * np.inf

        # set observation space
        self.observation_space = gym.spaces.Box(
            low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32
        )

        # set action space
        self.action_space = gym.spaces.Box(
            low=np.array([-1,-1,0]),
            high=np.array([1, 1,1]),
            dtype=np.float32,
        )

        self._task = task
        self.tasks = self.sample_tasks(n_tasks)
        self._goal_vel = self.tasks[0].get('roll', 0.0)
        self._goal = self._goal_vel
        self.all_goals = [-1]

    
    def sample_tasks(self, num_tasks):
        np.random.seed(1337)
        velocities = np.zeros(num_tasks)#np.random.uniform(-0.5, 0.5, size=(num_tasks,))
        tasks = [{'roll': velocity} for velocity in velocities]

        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_vel = self._task['roll']
        self._goal = self._goal_vel
        print(self._goal, idx)
        self.reset()


    def reset(self, state=None, target=None, **sim_reset_kw):
        """
        Reset state of environment.
        :param state: (dict) set initial value of states to given value.
        :param target: (dict) set initial value of targets to given value.
        :return: ([float]) observation vector
        """
        # reset steps
        self.steps_count = 0

        self.simulator = PyFly("/home/user/anaconda3/lib/python3.8/site-packages/pyfly/pyfly_config.json", "/home/user/anaconda3/lib/python3.8/site-packages/pyfly/x8_param.mat")

        self.simulator.seed(0)
        self.simulator.reset(state={"roll": self._goal, "pitch": 0.0, "Wind": 1})
        #self.simulator.turbulence = True
        #self.simulator.turbulence_intensity = "light"
        
        
        obs = self.get_observation()
        
        print('--reset--')

        return obs

    def step(self, action):
        """
        Perform one step from action chosen by agent.
        :param action: ([float]) the action chosen by the agent
        :return: ([float], float, bool, dict) observation vector, reward, done, extra information about episode on done
        """
        # check if any action is nan
        assert not np.any(np.isnan(action))

        #action[2] = abs(action[2])

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
            if self.rec:
                self.rec.savestate(
                    self.simulator.state, self.simulator.cur_sim_step -1
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
        info = {}
       
        return obs, reward, done, info

    def sample_target(self):
        self.target = {}
        self._target_props = {}
        self.cfg["observation"]['noise'] = None
        self._steps_for_current_target = 0 
        import sys
        sys.path.append('../libs/pyfly-fixed-wing-visualizer/')
        from pyfly_fixed_wing_visualizer.pyfly_fixed_wing_visualizer import simrecorder

        self.rec = simrecorder(self.steps_max)
        self.target['roll'] = 0
        self.target['pitch'] = 0
        self.target['Va'] = 22
        self.target['position_n'] = 20

    

    def render(self, mode="plot", show=True, close=True, block=False, save_path=None, epoch = None):
        """
        Visualize environment history. Plots of action and reward can be enabled through configuration file.
        :param mode: (str) render mode, one of plot for graph representation and animation for 3D animation with blender
        :param show: (bool) if true, plt.show is called, if false the figure is returned
        :param close: (bool) if figure should be closed after showing, or reused for next render call
        :param block: (bool) block argument to matplotlib blocking script from continuing until figure is closed
        :param save_path (str) if given, render is saved to this path.
        :return: (matplotlib Figure) if show is false in plot mode, the render figure is returned
        """
        if epoch == None:
            self.rec.plot(render = 'other')
        else:
            self.rec.plot(render = 'other',epoch = epoch)
            
    def get_reward(self, action=None, success=False, potential=False):
        """
        Get the reward for the current state of the environment.
        :return: (float) reward
        """
        # used
        reward = 0

       # for idx, meas in enumerate(self.rwd_vec):
       #     if self.targets[idx] != 0:
       #         val = -0.02* ((self.simulator.state[meas].value - self.targets[idx])/(self.targets[idx]))**2 #Always fly to 100,0,-50
       #     else:
       #         val = -0.02* (self.simulator.state[meas].value - self.targets[idx])**2 #Always fly to 100,0,-50
                
       #     reward += val
        curr_hgt = -1 * self.simulator.state['position_d'].value
        last_hgt = -1 * self.simulator.state['position_d'].history[-2]

        # negatives delta zum vorherigen zeitschritt
        #

        reward = curr_hgt - last_hgt
        #reward -= np.sum(np.asarray(action)**2)   
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




import datetime
#TODO: immer 12 AM
filename = 'Msac__' + datetime.date.today().strftime("%H_%M%p__%B_%d_%Y") + 'simpol.txt'


env = FixedWingAircraft_simple(config_path = "../libs/pyfly/pyfly/pyfly_config.json", n_tasks=130)

meta_model = mSAC('MlpPolicy', env, n_traintasks=100,n_evaltasks=30,n_epochtasks=5,verbose=1,policy_kwargs=dict(net_arch=[300, 300, 300], latent_dim = 5, hidden_sizes=[200,200,200]))#,learning_rate=0.0006)

meta_reward = []
meta_std = []

print('-Start-')
n_eval =30

meta_model_mean_reward_before, meta_model_std_reward_before = evaluate_meta_policy(meta_model, env, n_eval_episodes=n_eval, epoch=0)
meta_reward.append(meta_model_mean_reward_before)
meta_std.append(meta_model_std_reward_before)





my_file = open(filename,"w+")
with open(filename, "a") as myfile:
    myfile.write('epoch:'+ str(0)+"\n")
    myfile.write('meta_reward = '+ str(meta_reward)+"\n")
    myfile.write('meta_std = '+ str(meta_std)+"\n")
    myfile.write('================================================================================================\n\n')

print('##################################Start Learning##################################')
for i in range(50):
    
    meta_model.learn(total_timesteps=5*500)#, eval_freq=100, n_eval_episodes=5)
    meta_model_mean_reward, meta_model_std_reward = evaluate_meta_policy(meta_model, env, n_eval_episodes=n_eval, epoch=i+1)

    meta_reward.append(meta_model_mean_reward)
    meta_std.append(meta_model_std_reward)
    
    
    print('epoch:', i)
    print('meta_reward = ', meta_reward)
    print('meta_std = ', meta_std)
    

    with open(filename, "a") as myfile:
        myfile.write('epoch:'+ str(i+1)+"\n")
        myfile.write('meta_reward = '+ str(meta_reward)+"\n")
        myfile.write('meta_std = '+ str(meta_std)+"\n")
        myfile.write('================================================================================================\n\n')
env.close()

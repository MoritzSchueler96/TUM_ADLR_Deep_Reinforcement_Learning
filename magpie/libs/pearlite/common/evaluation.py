from typing import Callable, List, Optional, Tuple, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import base_class
from stable_baselines3.common.vec_env import VecEnv


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = env.reset()
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            action, state = model.predict(obs, state=state, deterministic=deterministic)
            obs, reward, done, _info = env.step(action)
            episode_reward += reward
            if callback is not None:
                callback(locals(), globals())
            episode_length += 1
            if render:
                env.render()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths
    return mean_reward, std_reward
    
    
    
def evaluate_meta_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    This is made to work only with one env.

    :param model: The RL agent you want to evaluate.
    :param env: The gym environment. In the case of a ``VecEnv``
        this must contain only one environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of reward per episode
        will be returned instead of the mean.
    :return: Mean reward per episode, std of reward per episode
        returns ([float], [int]) when ``return_episode_rewards`` is True
    """

    #assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

    _last_obs = []
    total_episodes = 0
    replay_buffer = model.replay_buffer
    continue_training = False
    num_timesteps = 0
    total_steps = 0
    if isinstance(env, VecEnv):
        assert env.num_envs == 1, "You must pass only one environment when using this function"

    episode_rewards, episode_lengths = [], []
    reward = []
    for i in range(n_eval_episodes):
        # Avoid double reset, as VecEnv are reset automatically
        if not isinstance(env, VecEnv) or i == 0:
            obs = th.Tensor(env.reset())
            _last_obs = obs
        done, state = False, None
        episode_reward = 0.0
        episode_length = 0
        while not done:
            with th.no_grad():

                if replay_buffer.pos > 999:
                    #print('z schould change')
                    context = model.actor.sample_context(replay_buffer)
                    #print(replay_buffer.observations.shape)
                    model.actor.infer_posterior(context)
                    
                    model.actor.sample_z()
                # Select action randomly or according to policy
                
                action, buffer_action = model.actor.predict(obs, model.actor.z, deterministic=True) #none for deterministic
                
                
                # Rescale and perform action   
                new_obs, reward, done, infos = env.step(action)
                obs = th.Tensor(new_obs)
                
                    
               
                if model._vec_normalize_env is not None: ##delete? TODO
                    new_obs_ = model._vec_normalize_env.get_original_obs()
                    reward_ = model._vec_normalize_env.get_original_reward()

                else:
                    # Avoid changing the original ones
                    _last_original_obs, new_obs_, reward_ = _last_obs, new_obs, reward

                # Only stop training if return value is False, not when it is None.
    
                episode_reward += reward


                replay_buffer.add(obs = _last_original_obs, next_obs = new_obs, action  =action, reward =  reward_, done = done)
    
                _last_obs = new_obs
                # Save the unnormalized observation
          


        if done:
            total_episodes += 1
            #episode_num += 1
            episode_rewards.append(episode_reward)
            #total_timesteps.append(episode_timesteps)

            # Log training infos
           # if log_interval is not None and self._episode_num % log_interval == 0:
           #     self._dump_logs()
    std_reward = np.std(episode_rewards) if total_episodes > 0 else 0.0
    mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0
    return mean_reward, std_reward

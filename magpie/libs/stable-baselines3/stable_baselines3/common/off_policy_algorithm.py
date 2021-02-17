import io
import pathlib
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common import logger
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.vec_env import VecEnv


class OffPolicyAlgorithm(BaseAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Set to `-1` to disable.
    :param gradient_steps: How many gradient steps to do after each rollout
        (see ``train_freq`` and ``n_episodes_rollout``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param n_episodes_rollout: Update the model every ``n_episodes_rollout`` episodes.
        Note that this cannot be used at the same time as ``train_freq``. Set to `-1` to disable.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: Whether the model support gSDE or not
    :param remove_time_limit_termination: Remove terminations (dones) that are due to time limit.
        See https://github.com/hill-a/stable-baselines/issues/863
    """

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        policy_base: Type[BasePolicy],
        learning_rate: Union[float, Callable],
        buffer_size: int = int(1e6),
        learning_starts: int = 100,
        n_traintasks: int = 0,
        n_evaltasks: int = 0,
        n_epochtasks: int = 0,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        n_episodes_rollout: int = -1,
        action_noise: Optional[ActionNoise] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        remove_time_limit_termination: bool = False,
    ):

        super(OffPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.n_episodes_rollout = n_episodes_rollout
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage

        # Remove terminations (dones) that are due to time limit
        # see https://github.com/hill-a/stable-baselines/issues/863
        self.remove_time_limit_termination = remove_time_limit_termination

        if train_freq > 0 and n_episodes_rollout > 0:
            warnings.warn(
                "You passed a positive value for `train_freq` and `n_episodes_rollout`."
                "Please make sure this is intended. "
                "The agent will collect data by stepping in the environment "
                "until both conditions are true: "
                "`number of steps in the env` >= `train_freq` and "
                "`number of episodes` > `n_episodes_rollout`"
            )

        self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup
        print("###############################################")
        print("###############     CONFIG     ################")
        print("######## n_traintasks: ", n_traintasks, "###########")
        print("######## n_evaltasks: ", n_evaltasks, "###########")
        print("######## n_epochtasks: ", n_epochtasks, "###########")
        print("###############################################")

        self.n_traintasks = n_traintasks
        self.n_evaltasks = n_evaltasks
        self.n_epochtasks = n_epochtasks
        self._n_env_steps_total = 0
        self.initial_experience = False

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        self.RBList_replay = [None] * self.n_traintasks
        self.RBList_encoder = [None] * self.n_traintasks
        self.RBList_eval = [None] * self.n_evaltasks

        for i in range(self.n_traintasks):
            self.RBList_replay[i] = ReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
            )

        for i in range(self.n_traintasks):
            self.RBList_encoder[i] = ReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
            )

        for i in range(self.n_evaltasks):
            self.RBList_eval[i] = ReplayBuffer(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
            )

        base_length = 500

        self.num_initial_steps = 10 * base_length
        self.num_train_steps_per_itr = 2000
        self.num_steps_prior = 2 * base_length
        self.num_steps_posterior = 0
        self.num_extra_rl_steps_posterior = 3 * base_length
        self.update_post_train = 1
        self.num_iterations = 500
        self.num_tasks_sample = 5
        self.max_path_length = 1500
        self.train_tasks = 5
        self.meta_batch = 16
        self._n_train_steps_total = 0

        from stable_baselines3.common.preprocessing import get_action_dim

        self.act_dim = get_action_dim(self.action_space)

        self.obs_dim = self.observation_space.shape[0]

        self.JUST_EVAL = ReplayBuffer(
            self.buffer_size,
            self.observation_space,
            self.action_space,
            self.device,
            optimize_memory_usage=self.optimize_memory_usage,
        )

        self.callback = print("")

        #       self.replay_buffer = ReplayBuffer(
        #          self.buffer_size,
        #         self.observation_space,
        #        self.action_space,
        #       self.device,
        #      optimize_memory_usage=self.optimize_memory_usage,
        # )
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

    def save_replay_buffer(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase]
    ) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self, path: Union[str, pathlib.Path, io.BufferedIOBase]
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(
            self.replay_buffer, ReplayBuffer
        ), "The replay buffer must inherit from ReplayBuffer class"

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: Union[None, Callable, List[BaseCallback], BaseCallback] = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        cf `BaseAlgorithm`.
        """
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46
        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and self.replay_buffer is not None
            and (self.replay_buffer.full or self.replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (self.replay_buffer.pos - 1) % self.replay_buffer.buffer_size
            self.replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, self.callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )
        callback = self.callback
        callback.on_training_start(locals(), globals())

        """
        meta-training loop
        """

        for it_ in range(1):
            if it_ == 0 and self.initial_experience == False:
                self.actor.z_means = th.zeros(5)
                self.actor.z_vars = th.ones(5)
                self.actor.sample_z()
                self.actor.context = None

                print("collecting initial pool of data for train and eval")
                # temp for evaluating
                for idx in range(self.n_traintasks):
                    self.task_idx = idx
                    self.env.env_method("reset_task", self.task_idx)
                    self.collect_data(self.num_initial_steps, 1, np.inf)
                self.initial_experience = True
            # Sample data from train tasks.
            else:
                print(self.RBList_replay[1].pos)
            for i in range(self.num_tasks_sample):
                idx = np.random.randint(self.n_traintasks)
                self.task_idx = idx
                self.env.envs[0].reset_task(idx)
                self.RBList_encoder[idx].reset()

                # collect some trajectories with z ~ prior
                if self.num_steps_prior > 0:
                    self.collect_data(self.num_steps_prior, 1, np.inf)
                # collect some trajectories with z ~ posterior
                if self.num_steps_posterior > 0:
                    self.collect_data(
                        self.num_steps_posterior, 1, self.update_post_train
                    )
                # even if encoder is trained only on samples from the prior, the policy needs to learn to handle z ~ posterior
                if self.num_extra_rl_steps_posterior > 0:
                    self.collect_data(
                        self.num_extra_rl_steps_posterior,
                        1,
                        self.update_post_train,
                        add_to_enc_buffer=False,
                    )

            # Sample train tasks and compute gradient updates on parameters.
            print("apply grads")
            for train_step in range(self.num_train_steps_per_itr):
                indices = np.random.choice(self.n_traintasks, self.meta_batch)
                self._do_training(indices)
                self._n_train_steps_total += 1

                print("----- ", train_step, " -----")

            self._dump_logs()
            self.ent_coef_losses, self.ent_coefs = [], []
            self.actor_losses, self.critic_losses = [], []
            self.kl_losses = []
            self.l_z_means, self.l_z_vars = [], []

        callback.on_training_end()

    def collect_data(
        self,
        num_samples,
        resample_z_rate,
        update_posterior_rate,
        add_to_enc_buffer=True,
    ):
        """
        get trajectories from current env in batch mode with given policy
        collect complete trajectories until the number of collected transitions >= num_samples

        :param agent: policy to rollout
        :param num_samples: total number of transitions to sample
        :param resample_z_rate: how often to resample latent context z (in units of trajectories)
        :param update_posterior_rate: how often to update q(z | c) from which z is sampled (in units of trajectories)
        :param add_to_enc_buffer: whether to add collected data to encoder replay buffer
        """
        # start from the prior
        self.actor.clear_z()

        num_transitions = 0
        while num_transitions < num_samples:
            if add_to_enc_buffer:
                n_samples = self.obtain_samples(
                    max_samples=num_samples - num_transitions,
                    max_trajs=update_posterior_rate,
                    accum_context=False,
                    resample=resample_z_rate,
                    replaybuffers=[
                        self.RBList_replay[self.task_idx],
                        self.RBList_encoder[self.task_idx],
                    ],
                )
            else:
                n_samples = self.obtain_samples(
                    max_samples=num_samples - num_transitions,
                    max_trajs=update_posterior_rate,
                    accum_context=False,
                    resample=resample_z_rate,
                    replaybuffers=[self.RBList_replay[self.task_idx]],
                )

            if update_posterior_rate != np.inf:
                context = self.sample_context(self.task_idx)
                self.actor.infer_posterior(context)

            num_transitions += n_samples

        self._n_env_steps_total += num_transitions

    def obtain_samples(
        self,
        deterministic=False,
        max_samples=np.inf,
        max_trajs=np.inf,
        accum_context=True,
        resample=1,
        replaybuffers=[],
    ):
        """
        Obtains samples in the environment until either we reach either max_samples transitions or
        num_traj trajectories.
        The resample argument specifies how often (in trajectories) the agent will resample it's context.
        """
        assert (
            max_samples < np.inf or max_trajs < np.inf
        ), "either max_samples or max_trajs must be finite"

        n_steps_total = 0
        n_trajs = 0
        while n_steps_total < max_samples and n_trajs < max_trajs:
            # save the latent context that generated this trajectory
            rollout = self.collect_rollouts(
                self.env,
                n_episodes=1,
                n_steps=self.max_path_length,
                action_noise=self.action_noise,
                callback=self.callback,
                learning_starts=self.learning_starts,
                replay_buffer=replaybuffers,
                log_interval=None,
                accum_context=accum_context,
            )
            n_steps_total += self.max_path_length
            n_trajs += 1
            # don't we also want the option to resample z ever transition?
            if n_trajs % resample == 0:
                self.actor.sample_z()
        return n_steps_total

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (
            self.use_sde and self.use_sde_at_warmup
        ):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter

            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        fps = int(self.num_timesteps / (time.time() - self.start_time))
        logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            logger.record(
                "rollout/ep_rew_mean",
                safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]),
            )
            logger.record(
                "rollout/ep_len_mean",
                safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]),
            )
        logger.record("time/fps", fps)
        logger.record(
            "time/time_elapsed",
            int(time.time() - self.start_time),
            exclude="tensorboard",
        )
        logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            logger.record("rollout/success rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        logger.dump(step=self.num_timesteps)

    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_episodes: int = 1,
        n_steps: int = -1,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        replay_buffer: Optional[List[ReplayBuffer]] = None,
        log_interval: Optional[int] = None,
        accum_context: bool = False,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ReplayBuffer.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param n_episodes: Number of episodes to use to collect rollout data
            You can also specify a ``n_steps`` instead
        :param n_steps: Number of steps to use to collect rollout data
            You can also specify a ``n_episodes`` instead.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        episode_rewards, total_timesteps = [], []
        total_steps, total_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"

        if self.use_sde:
            self.actor.reset_noise()
        try:
            callback.on_rollout_start()
        except:
            pass
        continue_training = True
        o = env.reset()
        next_o = None
        while total_steps < n_steps or total_episodes < n_episodes:
            done = False
            episode_reward, episode_timesteps = 0.0, 0

            while not done:

                if (
                    self.use_sde
                    and self.sde_sample_freq > 0
                    and total_steps % self.sde_sample_freq == 0
                ):
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                a, agent_info = self.actor.get_action(th.Tensor(o).reshape(-1, 1, 1))

                # disable for mujoco:

                # Rescale and perform action
                next_o, reward, done, infos = env.step([a])
                if accum_context:
                    self.actor.update_context([o, a, reward, next_o, done, infos])

                self.num_timesteps += 1
                episode_timesteps += 1
                total_steps += 1

                # Give access to local variables
                try:
                    callback.update_locals(locals())
                    # Only stop training if return value is False, not when it is None.
                    if callback.on_step() is False:
                        return RolloutReturn(
                            0.0, total_steps, total_episodes, continue_training=False
                        )
                except:
                    pass

                episode_reward += reward

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer
                if replay_buffer is not None:
                    for RB in replay_buffer:
                        RB.add(
                            obs=o, next_obs=next_o, action=a, reward=reward, done=done
                        )

                o = next_o  # Save the unnormalized observation

                self._update_current_progress_remaining(
                    self.num_timesteps, self._total_timesteps + 1
                )

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                if 0 < n_steps <= total_steps:
                    total_episodes += 1
                    self._episode_num += 1
                    episode_rewards.append(episode_reward)
                    total_timesteps.append(episode_timesteps)
                    break

            if done:
                total_episodes += 1
                self._episode_num += 1
                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()

        mean_reward = np.mean(episode_rewards) if total_episodes > 0 else 0.0
        try:
            callback.on_rollout_end()
        except:
            pass
        return RolloutReturn(
            mean_reward, total_steps, total_episodes, continue_training
        )

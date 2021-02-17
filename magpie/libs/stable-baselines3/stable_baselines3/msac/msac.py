from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common import logger
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.msac.policies import SACPolicy


class mSAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
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
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Callable] = 3e-4,
        buffer_size: int = int(1e6),
        learning_starts: int = 0,
        
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
        ent_coef: Union[str, float] = "auto",
        target_update_interval: int = 1,
        target_entropy: Union[str, float] = "auto",
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        policy_kwargs: Dict[str, Any] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):

        super(mSAC, self).__init__(
            policy,
            env,
            SACPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            n_traintasks,
            n_evaltasks,
            n_epochtasks,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            n_episodes_rollout,
            action_noise,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        self.indices = None
        self.context = None
        
        self.encodermap = None
        self.replaymap = None
        
        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super(mSAC, self)._setup_model()
        self._create_aliases()
        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = th.log(th.ones(1, device=self.device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = th.optim.Adam([self.log_ent_coef], lr=self.lr_schedule(1))
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef_tensor = th.tensor(float(self.ent_coef)).to(self.device)

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target
        
    def sample_context(self, indices, buff = None):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
#        print('indices in contextsampling', indices)

        
        final = th.zeros(len(indices),100,27)
        
      
   #     print(indices)
    #    print(self.encodermap['end'])
     #   print(self.replaymap['end'])
        if len(indices) >1:
            
            for i,idx in enumerate(indices):
        
                sample_pos = np.random.randint(self.RBList_encoder[idx].pos, size=(100))
 
                sample = self.RBList_encoder[idx]._get_samples(sample_pos) 

                final[i]=th.cat([sample.observations,sample.actions,sample.rewards], dim=1)
        else:
            if buff is not None:
                sample_pos = np.random.randint(buff[indices[0]].pos, size=(100))
 
                sample = buff[indices[0]]._get_samples(sample_pos) 

                final=th.cat([sample.observations,sample.actions,sample.rewards], dim=1)           
            
            else:
                sample_pos = np.random.randint(self.RBList_encoder[indices[0]].pos, size=(100))
 
                sample = self.RBList_encoder[indices[0]]._get_samples(sample_pos) 

                final=th.cat([sample.observations,sample.actions,sample.rewards], dim=1)
            final = final.view(1, 100, 27)
        return final
        
    def sample_context_atinference(self, indices=[], depth =200):
        ''' sample batch of context from a list of tasks from the replay buffer '''
        # make method work given a single task index
        if not hasattr(indices, '__iter__'):
            indices = [indices]
#        print('indices in contextsampling', indices)

        final = th.zeros(1,100,27)
        
        if depth == 0:
            depth = 1

        sample_pos = np.random.randint(depth, size=(100)) #FUCKING TODO
           
        sample = self.replay_buffer._get_samples(self.replay_buffer.pos - sample_pos)   

        final=th.cat([sample.observations,sample.actions,sample.rewards], dim=1)
        
        return final

    def train(self, gradient_steps: int, batch_size: int = 64, indices = []) -> None:
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.actor.context_optimizer, self.critic.optimizer] 
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []
        kl_losses = []
        l_z_means, l_z_vars = [], []
             
           
        self.indices = indices  
        
        for gradient_step in range(gradient_steps):
        
           
            
            
            self.context = self.sample_context(self.indices)#(gradient_steps-gradient_step))
#            print('performing gradient Step')
            # Sample replay buffer
            obs = th.zeros(16,batch_size,20)
            next_observations = th.zeros(16,batch_size,20)
            actions = th.zeros(16,batch_size,6)
            rewards = th.zeros(16,batch_size,1)
            dones = th.zeros(16,batch_size,1)

            for i,idx in enumerate(self.indices):
        
                sample_pos = np.random.randint(self.RBList_replay[idx].pos,size=(batch_size))
 
                sample = self.RBList_replay[idx]._get_samples(sample_pos) 


                obs[i]=sample.observations
                next_observations[i]=sample.next_observations
                actions[i]=sample.actions
                rewards[i]=sample.rewards
                dones[i]=sample.dones
#            print('obs[0][0][0]',obs[0][0][0], obs.shape) 
#            print('obs[0][0][0]',next_observations[0][0][0], next_observations.shape) 
                   
            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()        
        
            # run inference in networks
            self.actor.infer_posterior(self.context)
            self.actor.sample_z()
            task_z = self.actor.z
            
#            print('z_before:', task_z)
            
            t, b, _ = obs.size()
#            print(t, b)
            obs = obs.view(t * b, -1)
            next_observations = next_observations.view(t * b, -1)
            rewards = rewards.view(t * b, -1)
            dones = dones.view(t * b, -1)
            actions = actions.view(t * b, -1)
            
            task_z = [z.repeat(b, 1) for z in task_z]
            task_z = th.cat(task_z, dim=0)
            
#            print('task_z',task_z, task_z.shape)
                         
            # logging
            local_means = self.actor.z_means.clone().detach().numpy()
            local_vars = self.actor.z_vars.clone().detach().numpy()
            l_z_means.append(local_means)
            l_z_vars.append(local_vars)
            
            
            # run policy, get log probs and new actions
            actions_pi, log_prob = self.actor.action_log_prob(obs, task_z.clone().detach())
            log_prob = log_prob.reshape(-1, 1)
#            print(actions_pi, actions_pi.shape)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
                
                
                
            # KL constraint on z if probabilistic
            self.actor.context_optimizer.zero_grad()
            kl_div = self.actor.compute_kl_div()
            kl_loss = 0.1 * kl_div
            kl_loss.backward(retain_graph=True)
            kl_losses.append(kl_loss.clone().detach().numpy())    
                

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(next_observations, task_z.clone().detach())
                # Compute the target Q value: min over all critics targets
                
                next_actions_and_z = th.cat([next_actions, task_z.clone().detach()], dim=1)
                targets = th.cat(self.critic_target(next_observations, next_actions_and_z), dim=1)
                target_q, _ = th.min(targets, dim=1, keepdim=True)
                # add entropy term
                
#                print(target_q.shape)
                
                target_q = target_q - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
#                print(target_q.shape)
#                print(rewards.shape)
#                print(dones.shape)
                
                q_backup = ( rewards * 5) + (1 - dones) * self.gamma * target_q

            # Q and V networks
            # encoder will only get gradients from Q nets
            # Get current Q estimates for each critic network
            # using action from the replay buffer
            actions_and_z = th.cat([actions, task_z], dim=1)
            current_q_estimates = self.critic(obs, actions_and_z)
           
            

            

            

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, q_backup) for current_q in current_q_estimates])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            self.critic.optimizer.step()
            self.actor.context_optimizer.step()

            #Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            own_actions_and_z = th.cat([actions_pi, task_z.clone().detach()], dim=1)
            q_values_pi = th.cat(self.critic.forward(obs, own_actions_and_z), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()
            
            self.actor.detach_z()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        logger.record("train/ent_coef", np.mean(ent_coefs))
        logger.record("train/actor_loss", np.mean(actor_losses))
        logger.record("train/critic_loss", np.mean(critic_losses))
        logger.record("train/KL_loss", np.mean(np.asarray(kl_losses)))
        logger.record("train/avg. z", np.mean(np.asarray(l_z_means)))
        logger.record("train/avg. z var", np.mean(np.asarray(l_z_vars)))
        if len(ent_coef_losses) > 0:
            logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "mSAC",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
    ) -> OffPolicyAlgorithm:

        return super(mSAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(mSAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        saved_pytorch_variables = ["log_ent_coef"]
        if self.ent_coef_optimizer is not None:
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables.append("ent_coef_tensor")
        return state_dicts, saved_pytorch_variables
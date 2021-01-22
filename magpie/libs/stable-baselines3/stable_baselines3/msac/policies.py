from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import gym
import torch as th
from torch import nn
import torch.nn.functional as F
import numpy as np

from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution, StateDependentNoiseDistribution
from stable_baselines3.common.policies import BasePolicy, ContinuousCritic, create_sde_features_extractor, register_policy
from stable_baselines3.common.preprocessing import get_action_dim
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
    get_actor_critic_arch,
)

# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20

def _product_of_gaussians(mus, sigmas_squared):
    '''
    compute mu, sigma of product of gaussians
    '''
    sigmas_squared = th.clamp(sigmas_squared, min=1e-7)
    sigma_squared = 1. / th.sum(th.reciprocal(sigmas_squared), dim=0)
    mu = sigma_squared * th.sum(mus / sigmas_squared, dim=0)
    return mu, sigma_squared

class Actor(BasePolicy):
    """
    Actor network (policy) for SAC.

    :param observation_space: Obervation space
    :param action_space: Action space
    :param net_arch: Network architecture
    :param features_extractor: Network to extract features
        (a CNN when using images, a nn.Flatten() layer otherwise)
    :param features_dim: Number of features
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param full_std: Whether to use (n_features x n_actions) parameters
        for the std instead of only (n_features,) when using gSDE.
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        net_arch: List[int],
        features_extractor: nn.Module,
        features_dim: int,
        latent_dim: int,
        hidden_sizes: List[int] = [200, 200, 200],
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        normalize_images: bool = True,
    ):
        super(Actor, self).__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
            squash_output=True,
        )

        # Save arguments to re-create object at loading
        self.use_sde = use_sde
        self.sde_features_extractor = None
        self.sde_net_arch = sde_net_arch
        self.net_arch = net_arch
        self.features_dim = features_dim
        self.activation_fn = activation_fn
        self.log_std_init = log_std_init
        self.sde_net_arch = sde_net_arch
        self.use_expln = use_expln
        self.full_std = full_std
        self.clip_mean = clip_mean
        
        self.latent_dim = latent_dim
        self.observation_space = observation_space
        action_dim = get_action_dim(self.action_space)
        latent_pi_net = create_mlp(features_dim+latent_dim, -1, net_arch, activation_fn)
        self.latent_pi = nn.Sequential(*latent_pi_net)
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else features_dim
        
        # create context encoder network
        reward_dim = 1
        #context_encoder_input_dim = 2*features_dim + action_dim + reward_dim
        context_encoder_input_dim = features_dim + action_dim + reward_dim
        self.context_encoder_output_dim = latent_dim * 2
        context_encoder = create_mlp(input_dim = context_encoder_input_dim, output_dim = self.context_encoder_output_dim, net_arch = hidden_sizes, activation_fn = nn.ReLU)
        self.context_encoder = nn.Sequential(*context_encoder).to(self.device)
        self.latent_dim = latent_dim
        

        if self.use_sde:
            latent_sde_dim = last_layer_dim
            # Separate features extractor for gSDE
            if sde_net_arch is not None:
                self.sde_features_extractor, latent_sde_dim = create_sde_features_extractor(
                    features_dim, sde_net_arch, activation_fn
                )

            self.action_dist = StateDependentNoiseDistribution(
                action_dim, full_std=full_std, use_expln=use_expln, learn_features=True, squash_output=True
            )
            self.mu, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=last_layer_dim, latent_sde_dim=latent_sde_dim, log_std_init=log_std_init
            )
            # Avoid numerical issues by limiting the mean of the Gaussian
            # to be in [-clip_mean, clip_mean]
            if clip_mean > 0.0:
                self.mu = nn.Sequential(self.mu, nn.Hardtanh(min_val=-clip_mean, max_val=clip_mean))
        else:
            self.action_dist = SquashedDiagGaussianDistribution(action_dim)
            self.mu = nn.Linear(last_layer_dim, action_dim)
            self.log_std = nn.Linear(last_layer_dim, action_dim)
            
        self.z_means = th.zeros(latent_dim)
        self.z_vars = th.ones(latent_dim)
        self.z = th.zeros(latent_dim)
        
        
    def clear_z(self):
        '''
        reset q(z|c) to the prior
        sample a new z from the prior
        '''
        # reset distribution over z to the prior
        mu = th.zeros(self.latent_dim)
        
        var = th.ones(self.latent_dim)

        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
#        print('z cleared')

    def sample_z(self):
        
        posteriors = [th.distributions.Normal(m, th.sqrt(s)) for m, s in zip(th.unbind(self.z_means), th.unbind(self.z_vars))]
        z = [d.rsample() for d in posteriors]
        self.z = th.stack(z)
#        self.z = th.zeros_like(self.z)
        
    def detach_z(self):
        ''' disable backprop through z '''
        self.z = self.z.detach()

    def compute_kl_div(self):
        ''' compute KL( q(z|c) || r(z) ) '''
        prior = th.distributions.Normal(th.zeros(self.latent_dim), th.ones(self.latent_dim))
        posteriors = [th.distributions.Normal(mu, th.sqrt(var)) for mu, var in zip(th.unbind(self.z_means), th.unbind(self.z_vars))]
        kl_divs = [th.distributions.kl.kl_divergence(post, prior) for post in posteriors]
        kl_div_sum = th.sum(th.stack(kl_divs))
        return kl_div_sum

    def infer_posterior(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)

        params = params.view(context.size(0), -1, 2*self.latent_dim)
        # with probabilistic z, predict mean and variance of q(z | c)
       
        mu = params[..., :self.latent_dim]
        
        
        sigma_squared = F.softplus(params[..., self.latent_dim:])
        z_params = [_product_of_gaussians(m, s) for m, s in zip(th.unbind(mu), th.unbind(sigma_squared))]

        self.z_means = th.stack([p[0] for p in z_params])
        self.z_vars = th.stack([p[1] for p in z_params])
       
        
        self.sample_z()
        
    def infer_posterior_atinference(self, context):
        ''' compute q(z|c) as a function of input context and sample new z from it'''
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, 2*self.latent_dim)
        # with probabilistic z, predict mean and variance of q(z | c)
       
        mu = params[..., :self.latent_dim]
        
        mu = th.mean(mu,dim=0).reshape(self.latent_dim,1)

        sigma_squared = F.softplus(params[..., self.latent_dim:])
        
        sigma_squared = th.mean(sigma_squared,dim=0).reshape(self.latent_dim,1)
        
        
        
        z_params = [_product_of_gaussians(m, s) for m, s in zip(th.unbind(mu), th.unbind(sigma_squared))]



        self.z_means = th.stack([p[0] for p in z_params])
        self.z_vars = th.stack([p[1] for p in z_params])
        
        self.sample_z()

       

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                use_sde=self.use_sde,
                log_std_init=self.log_std_init,
                full_std=self.full_std,
                sde_net_arch=self.sde_net_arch,
                use_expln=self.use_expln,
                features_extractor=self.features_extractor,
                clip_mean=self.clip_mean,
            )
        )
        return data

    def get_std(self) -> th.Tensor:
        """
        Retrieve the standard deviation of the action distribution.
        Only useful when using gSDE.
        It corresponds to ``th.exp(log_std)`` in the normal case,
        but is slightly different when using ``expln`` function
        (cf StateDependentNoiseDistribution doc).

        :return:
        """
        msg = "get_std() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        return self.action_dist.get_std(self.log_std)

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        msg = "reset_noise() is only available when using gSDE"
        assert isinstance(self.action_dist, StateDependentNoiseDistribution), msg
        self.action_dist.sample_weights(self.log_std, batch_size=batch_size)

    def get_action_dist_params(self, obs: th.Tensor, z: th.Tensor, other_shape: bool = False) -> Tuple[th.Tensor, th.Tensor, Dict[str, th.Tensor]]:
        """
        Get the parameters for the action distribution.

        :param obs:
        :return:
            Mean, standard deviation and optional keyword arguments.
        """
#        features = self.extract_features(obs)
        features = obs
        #if int(features.shape[0]) == 1:
            #features = features.flatten()
        if len(z.shape) == 1:
            z = z.reshape(1, self.latent_dim)

        #if int(z.shape[0]) == 1 and int(z.shape[1]) == self.latent_dim:
            #z = z.flatten()

        
        features = th.cat([features, z], dim=1)
#        print('combined features:', features.shape)
        latent_pi = self.latent_pi(features)
        mean_actions = self.mu(latent_pi)

        if self.use_sde:
            latent_sde = latent_pi
            if self.sde_features_extractor is not None:
                latent_sde = self.sde_features_extractor(features)
            return mean_actions, self.log_std, dict(latent_sde=latent_sde)
        # Unstructured exploration (Original implementation)
        log_std = self.log_std(latent_pi)
        # Original Implementation to cap the standard deviation
        log_std = th.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        
  
#        print('MA Before:', mean_actions)
        #TODO: WHY?  
        #if not other_shape:      
#        print(mean_actions.shape)
        if mean_actions.shape[0] > 1:
            pass
        else:
            pass
        	#mean_actions = mean_actions.reshape(self.action_space.shape[0],1)
#        print('MA AFTER:', mean_actions)
        
#        print('observation.shape[1:]:',obs.shape[1:],' == ', 'observation_space.shape:', self.observation_space.shape)

        return mean_actions, log_std, {}

    def forward(self, obs: th.Tensor, z: th.Tensor = None, deterministic: bool = False, other_shape:bool = False) -> th.Tensor:
        """
        (a|z)
        """
        
        if z == None:
            self.sample_z()
            z = self.z
        #print(z)
        
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs,z, other_shape = other_shape)
        # Note: the action is squashed
#        print('MA AFTER:', mean_actions)
        return self.action_dist.actions_from_params(mean_actions, log_std, deterministic=deterministic, **kwargs)

    def action_log_prob(self, obs: th.Tensor, z: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
                
        mean_actions, log_std, kwargs = self.get_action_dist_params(obs, z)
        # return action and associated log prob
        return self.action_dist.log_prob_from_params(mean_actions, log_std, **kwargs)

    def _predict(self, obs: th.Tensor, z: th.Tensor = None, deterministic: bool = False) -> th.Tensor:

#        print('predict')
        return self.forward(obs, z, deterministic)


class SACPolicy(BasePolicy):
    """
    Policy class (with both actor and critic) for SAC.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param use_sde: Whether to use State Dependent Exploration or not
    :param log_std_init: Initial value for the log standard deviation
    :param sde_net_arch: Network architecture for extracting features
        when using gSDE. If None, the latent features from the policy will be used.
        Pass an empty list to use the states as features.
    :param use_expln: Use ``expln()`` function instead of ``exp()`` when using gSDE to ensure
        a positive standard deviation (cf paper). It allows to keep variance
        above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
    :param clip_mean: Clip the mean output when using gSDE to avoid numerical instability.
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    :param n_critics: Number of critic networks to create.
    :param share_features_extractor: Whether to share or not the features extractor
        between the actor and the critic (this saves computation time)
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable,
        latent_dim: int,
        hidden_sizes:  Optional[List[int]] = None,
        net_arch: Optional[Union[List[int], Dict[str, List[int]]]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        use_sde: bool = False,
        log_std_init: float = -3,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        clip_mean: float = 2.0,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        n_critics: int = 2,
        share_features_extractor: bool = True,
    ):
        super(SACPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=True,
        )

        if net_arch is None:
            if features_extractor_class == FlattenExtractor:
                net_arch = [256, 256]
            else:
                net_arch = []

        actor_arch, critic_arch = get_actor_critic_arch(net_arch)

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.net_args = {
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "net_arch": actor_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.actor_kwargs = self.net_args.copy()
        sde_kwargs = {
            "latent_dim": latent_dim,
            "use_sde": use_sde,
            "log_std_init": log_std_init,
            "sde_net_arch": sde_net_arch,
            "use_expln": use_expln,
            "clip_mean": clip_mean,
        }
        self.actor_kwargs.update(sde_kwargs)
        self.critic_kwargs = self.net_args.copy()
        
        
        self.critic_kwargs.update(
            {
                "n_critics": n_critics,
                "net_arch": critic_arch,
                "share_features_extractor": share_features_extractor,
            }
        )

        self.actor, self.actor_target = None, None
        self.critic, self.critic_target = None, None
        self.share_features_extractor = share_features_extractor

        self._build(lr_schedule)

    def _build(self, lr_schedule: Callable) -> None:
        self.actor = self.make_actor()
     
        
        self.actor.optimizer = self.optimizer_class(self.actor.mu.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)#TODO: actor.parameters??????
        self.actor.context_optimizer = self.optimizer_class(self.actor.context_encoder.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)

        if self.share_features_extractor:
            self.critic = self.make_critic(features_extractor=self.actor.features_extractor)
            # Do not optimize the shared features extractor with the critic loss
            # otherwise, there are gradient computation issues
            critic_parameters = [param for name, param in self.critic.named_parameters() if "features_extractor" not in name]
        else:
            # Create a separate features extractor for the critic
            # this requires more memory and computation
            self.critic = self.make_critic(features_extractor=None)
            critic_parameters = self.critic.parameters()

        # Critic target should not share the features extractor with critic
        self.critic_target = self.make_critic(features_extractor=None)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic.optimizer = self.optimizer_class(critic_parameters, lr=lr_schedule(1), **self.optimizer_kwargs)

    def _get_data(self) -> Dict[str, Any]:
        data = super()._get_data()

        data.update(
            dict(
                net_arch=self.net_arch,
                activation_fn=self.net_args["activation_fn"],
                use_sde=self.actor_kwargs["use_sde"],
                log_std_init=self.actor_kwargs["log_std_init"],
                sde_net_arch=self.actor_kwargs["sde_net_arch"],
                use_expln=self.actor_kwargs["use_expln"],
                clip_mean=self.actor_kwargs["clip_mean"],
                n_critics=self.critic_kwargs["n_critics"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def reset_noise(self, batch_size: int = 1) -> None:
        """
        Sample new weights for the exploration matrix, when using gSDE.

        :param batch_size:
        """
        self.actor.reset_noise(batch_size=batch_size)

    def make_actor(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> Actor:
        actor_kwargs = self._update_features_extractor(self.actor_kwargs, features_extractor)
        return Actor(**actor_kwargs).to(self.device)

    def make_critic(self, features_extractor: Optional[BaseFeaturesExtractor] = None) -> ContinuousCritic:
        critic_kwargs = self._update_features_extractor(self.critic_kwargs, features_extractor)
        critic_kwargs['features_dim'] = critic_kwargs['features_dim'] + self.actor.latent_dim
        return ContinuousCritic(**critic_kwargs).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = False) -> th.Tensor:
        print('this fwd?')
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, observation: th.Tensor, deterministic: bool = False) -> th.Tensor:
        #this is the predict fkt for 'evaluate'
#        print('predict?')
        action = self.actor(observation, self.actor.z, deterministic=deterministic, other_shape = True)
        
#        print('returned action', action.reshape(1,1,self.action_space.shape[0]))
        
        return action.reshape(1,self.action_space.shape[0])


MlpPolicy = SACPolicy

register_policy("MlpPolicy", MlpPolicy)


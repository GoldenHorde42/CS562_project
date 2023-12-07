# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # Actor-Critic Network
# class ActorCriticNetwork(nn.Module):
#     def __init__(self, state_dim, action_dim):
#         super(ActorCriticNetwork, self).__init__()
        
#         # Shared layers
#         self.shared_layers = nn.Sequential(
#             nn.Linear(state_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU()
#         )
#         self._init_weights(self.shared_layers)

#         # Actor layers for mean of actions
#         self.actor_mu = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim)  # Outputs mean for each action dimension
#         )
#         self._init_weights(self.actor_mu)

#         # Actor layers for standard deviation of actions
#         self.actor_sigma = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, action_dim)  # Outputs log standard deviation for each action dimension
#         )
#         self._init_weights(self.actor_sigma)

#         # Initialize the sigma to a small number to start with exploration
#         self.actor_sigma[-1].weight.data.fill_(0.0)
#         self.actor_sigma[-1].bias.data.fill_(-1.0)

#         # Critic layers
#         self.critic = nn.Sequential(
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)  # Single value output
#         )
#         self._init_weights(self.critic)
    
#     def forward(self, state):
#         x = self.shared_layers(state)
#         mu = self.actor_mu(x)

#         # Compute sigma in a more controlled way to avoid extremely large values
#         raw_sigma = self.actor_sigma(x)
#         sigma = torch.clamp(raw_sigma, min=-20, max=2)  # Clamping the raw output
#         sigma = torch.exp(sigma)  # Now applying exp
#         sigma = torch.clamp(sigma, min=1e-3, max=50)  # Clamping the exponential output

#         value = self.critic(x)

#         # Sanity checks for NaNs
#         assert not torch.isnan(mu).any(), "NaN detected in mu"
#         assert not torch.isnan(sigma).any(), "NaN detected in sigma"
#         assert not torch.isnan(value).any(), "NaN detected in value"

#         return mu, sigma, value

#     def act(self, state):
#         mu, sigma, _ = self.forward(state)
#         dist = torch.distributions.Normal(mu, sigma)
#         action = dist.sample()
#         action_logprob = dist.log_prob(action).sum(-1)
#         return action, action_logprob

#     def evaluate(self, state, action):
#         mu, sigma, value = self.forward(state)
#         dist = torch.distributions.Normal(mu, sigma)
#         action_logprob = dist.log_prob(action).sum(-1)
#         dist_entropy = dist.entropy().sum(-1)
#         return action_logprob, torch.squeeze(value), dist_entropy

#     def _init_weights(self, module):
#         for m in module.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_uniform_(m.weight)
#                 nn.init.zeros_(m.bias)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU()
        )
        self._init_weights(self.shared_layers)

        # Actor layers for mean of actions
        self.actor_mu = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Tanh activation to bound the actions
        )
        self._init_weights(self.actor_mu)

        # Parameter for standard deviation of actions
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic layers
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )
        self._init_weights(self.critic)
    
    def forward(self, state):
        x = self.shared_layers(state)
        mu = self.actor_mu(x)

        # Compute sigma using the parameterized log_std
        sigma = torch.exp(self.log_std).expand_as(mu)

        value = self.critic(x)
        assert not torch.isnan(mu).any(), "NaN detected in mu"
        assert not torch.isnan(sigma).any(), "NaN detected in sigma"
        assert not torch.isnan(value).any(), "NaN detected in value"

        return mu, sigma, value

    def act(self, state):
        mu, sigma, _ = self.forward(state)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        action_logprob = dist.log_prob(action).sum(-1)
        return action, action_logprob

    def evaluate(self, state, action):
        mu, sigma, value = self.forward(state)
        dist = torch.distributions.Normal(mu, sigma)
        action_logprob = dist.log_prob(action).sum(-1)
        dist_entropy = dist.entropy().sum(-1)
        return action_logprob, torch.squeeze(value), dist_entropy

    def _init_weights(self, module):
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self._init_weights(self.shared_layers)

        # Actor layers for mean of actions
        self.actor_mu = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # Outputs mean for each action dimension
        )
        self._init_weights(self.actor_mu)

        # Actor layers for standard deviation of actions
        self.actor_sigma = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # Outputs log standard deviation for each action dimension
        )
        self._init_weights(self.actor_sigma)

        # Initialize the sigma to a small number to start with exploration
        self.actor_sigma[-1].weight.data.fill_(0.0)
        self.actor_sigma[-1].bias.data.fill_(-1.0)

        # Critic layers
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )
        self._init_weights(self.critic)

    def forward(self, state):
        x = self.shared_layers(state)
        mu = self.actor_mu(x)
        sigma = torch.exp(self.actor_sigma(x).clamp(min=-20, max=2))  # Clamp to avoid extreme values
        value = self.critic(x)
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

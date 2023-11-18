import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor-Critic Network
class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim=9):  # Assuming 9 discrete actions
        super(ActorCriticNetwork, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Actor (policy) layers
        self.policy_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)  # Output layer for discrete actions
        )

        # Critic (value) layers
        self.value_layers = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Single value output
        )

    def forward(self, state):
        # Shared layers processing
        x = self.shared_layers(state)

        # Policy output (logits for discrete actions)
        policy_logits = self.policy_layers(x)

        # Value output
        value = self.value_layers(x)

        return policy_logits, value

    def policy(self, state):
        policy_logits = self.forward(state)[0]
        return F.softmax(policy_logits, dim=-1)

    def value(self, state):
        return self.forward(state)[1]

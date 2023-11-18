import torch
import torch.optim as optim

import torch
import torch.nn.functional as F
import torch.optim as optim

class PPOTrainer:
    def __init__(self, actor_critic, policy_lr, value_lr, gamma, eps_clip, k_epochs):
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs

        self.policy_optimizer = optim.Adam(actor_critic.policy_layers.parameters(), lr=policy_lr)
        self.value_optimizer = optim.Adam(actor_critic.value_layers.parameters(), lr=value_lr)

        self.memory = []

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        # Basic check for empty memory
        if not self.memory:
            raise ValueError("Memory is empty. Cannot train without data.")

        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.stack(states)
        actions = torch.stack(actions)
        rewards = torch.tensor(rewards)
        next_states = torch.stack(next_states)
        dones = torch.tensor(dones)

        # Check dimensions
        assert states.shape[0] == actions.shape[0] == rewards.shape[0] == next_states.shape[0] == dones.shape[0], \
            "Mismatch in dimensions of training data components."

        self.memory = []

        with torch.no_grad():
            old_probs = self.actor_critic.policy(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Policy training loop
        for _ in range(self.k_epochs):
            new_probs = self.actor_critic.policy(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            state_values = self.actor_critic.value(states).squeeze()
            next_state_values = self.actor_critic.value(next_states).squeeze()

            advantages = self.calculate_advantages(rewards, state_values, next_state_values, dones)

            ratios = new_probs / old_probs
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            value_loss = F.mse_loss(state_values, rewards + self.gamma * next_state_values * (1 - dones))

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def calculate_advantages(self, rewards, state_values, next_state_values, dones, gae_lambda=0.95):
        gae = 0
        advantages = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_state_values[step] * (1 - dones[step]) - state_values[step]
            gae = delta + self.gamma * gae_lambda * (1 - dones[step]) * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages)

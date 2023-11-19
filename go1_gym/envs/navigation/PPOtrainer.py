import torch
import torch.nn.functional as F
import torch.optim as optim

class PPOTrainer:
    def __init__(self, actor_critic, policy_lr, value_lr, gamma, eps_clip, k_epochs):
        self.actor_critic = actor_critic
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=policy_lr)
        self.memory = []

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train(self):
        if not self.memory:
            raise ValueError("Memory is empty. Cannot train without data.")

        states, actions, rewards, next_states, dones = zip(*self.memory)
        states = torch.stack(states).float()
        actions = torch.stack(actions).float()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.stack(next_states).float()
        dones = torch.tensor(dones, dtype=torch.float32)

        self.memory = []

        with torch.no_grad():
            _, old_logprobs, _ = self.actor_critic.evaluate(states, actions)
            next_state_values = self.actor_critic(next_states)[2].squeeze()

        for _ in range(self.k_epochs):
            logprobs, state_values, dist_entropy = self.actor_critic.evaluate(states, actions)
            advantages = self.calculate_advantages(rewards, state_values, next_state_values, dones)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            policy_loss = -torch.min(ratios * advantages, torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages).mean() - 0.01 * dist_entropy.mean()
            value_loss = F.mse_loss(state_values, rewards + self.gamma * next_state_values * (1 - dones))

            total_loss = policy_loss + value_loss
            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 1.0)  # Gradient clipping
            self.optimizer.step()

    def calculate_advantages(self, rewards, state_values, next_state_values, dones, gae_lambda=0.95):
        gae = 0
        advantages = []
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + self.gamma * next_state_values[step] * mask - state_values[step]
            gae = delta + self.gamma * gae_lambda * mask * gae
            advantages.insert(0, gae)
        
        return torch.tensor(advantages, dtype=torch.float32)

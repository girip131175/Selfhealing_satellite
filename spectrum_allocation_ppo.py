
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any

# Custom environment for spectrum allocation
class SpectrumAllocationEnv(gym.Env):
    def __init__(self, data_dir: str = "spectrum_demand_data", file_pattern: str = "spectrum_demand_*.csv"):
        super(SpectrumAllocationEnv, self).__init__()
        
        # Load all datasets
        self.datasets = self._load_datasets(data_dir, file_pattern)
        self.current_dataset_idx = 0
        self.current_step = 0
        self.df = self.datasets[self.current_dataset_idx]
        self.max_steps = len(self.df)
        
        # Identify features for 4G and 5G at low, mid, and high bands
        self.features = self._get_features()
        
        # Define state space: all metrics for 4G and 5G at different bands
        # Each technology-band combination has metrics like Demand, SNR, RSSI, etc.
        self.observation_space = spaces.Box(
            low=-float('inf'), 
            high=float('inf'), 
            shape=(len(self.features),)
        )
        
        # Define action space: allocation percentage for each band (low, mid, high) for both 4G and 5G
        # Actions are continuous values between 0 and 1 for each band's allocation proportion
        # Total allocation must sum to 1.0 for both 4G and 5G
        self.action_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(6,)  # [4G_low, 4G_mid, 4G_high, 5G_low, 5G_mid, 5G_high]
        )
        
        # Track metrics for evaluation
        self.total_reward = 0
        self.rewards_history = []

    def _load_datasets(self, data_dir: str, file_pattern: str) -> List[pd.DataFrame]:
        """Load all CSV datasets from the specified directory matching the pattern."""
        datasets = []
        import glob
        
        # Get list of all files matching the pattern
        file_paths = glob.glob(os.path.join(data_dir, file_pattern))
        
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            # Convert time to datetime if it exists
            if 'Time' in df.columns:
                df['Time'] = pd.to_datetime(df['Time'])
            datasets.append(df)
        
        return datasets
    
    def _get_features(self) -> List[str]:
        """Extract feature names from the dataset columns excluding 'Time'."""
        return [col for col in self.df.columns if col != 'Time']
    
    def _get_state(self) -> np.ndarray:
        """Get the current state (all metrics for the current timestep)."""
        if self.current_step >= len(self.df):
            return np.zeros(len(self.features))
        
        state = self.df.iloc[self.current_step][self.features].values
        # Normalize state values for better training stability
        return state
    
    def _calculate_reward(self, action: np.ndarray) -> float:
        """
        Calculate the reward based on the action taken.
        
        The reward function considers:
        1. Throughput satisfaction - how well the allocation meets the demand
        2. Quality of service - based on SNR and RSSI
        3. Efficiency - proper utilization of spectrum resources
        4. Fairness - distribution of resources among users
        """
        if self.current_step >= len(self.df):
            return 0
        
        state = self._get_state()
        
        # Extract metrics for calculation
        # This is a simplified approach - you may need to adjust based on your specific requirements
        
        # Create a dictionary for easier access to state values
        state_dict = dict(zip(self.features, state))
        
        # Normalize actions to ensure they sum to 1.0 for each technology
        action_4g = action[0:3]
        action_5g = action[3:6]
        
        action_4g = action_4g / np.sum(action_4g) if np.sum(action_4g) > 0 else action_4g
        action_5g = action_5g / np.sum(action_5g) if np.sum(action_5g) > 0 else action_5g
        
        # Calculate components of the reward
        throughput_reward = 0
        qos_reward = 0
        efficiency_reward = 0
        fairness_reward = 0
        
        # 1. Throughput satisfaction
        bands = ["Low", "Mid", "High"]
        for i, band in enumerate(bands):
            # 4G throughput
            demand_4g = state_dict.get(f"4G_{band}_Demand", 0)
            traffic_4g = state_dict.get(f"4G_{band}_Traffic", 0)
            allocated_4g = action_4g[i]
            
            # Calculate throughput satisfaction as ratio of traffic to demand
            if demand_4g > 0:
                satisfaction_4g = min(traffic_4g * allocated_4g / demand_4g, 1.0)
                throughput_reward += satisfaction_4g
            
            # 5G throughput
            demand_5g = state_dict.get(f"5G_{band}_Demand", 0)
            traffic_5g = state_dict.get(f"5G_{band}_Traffic", 0)
            allocated_5g = action_5g[i]
            
            if demand_5g > 0:
                satisfaction_5g = min(traffic_5g * allocated_5g / demand_5g, 1.0)
                throughput_reward += satisfaction_5g
        
        # 2. Quality of Service based on SNR and RSSI
        for i, band in enumerate(bands):
            # 4G QoS
            snr_4g = state_dict.get(f"4G_{band}_SNR", 0)
            rssi_4g = state_dict.get(f"4G_{band}_RSSI", 0)
            
            # Normalize SNR and RSSI for QoS calculation
            # Higher SNR is better, values typically range from 0 to 30
            snr_norm_4g = min(max(snr_4g / 30.0, 0), 1.0)
            
            # RSSI is typically negative, with values around -60 to -100 dBm
            # Less negative is better
            rssi_norm_4g = min(max((rssi_4g + 100) / 40.0, 0), 1.0)
            
            qos_4g = (snr_norm_4g + rssi_norm_4g) / 2.0
            qos_reward += qos_4g * action_4g[i]
            
            # 5G QoS
            snr_5g = state_dict.get(f"5G_{band}_SNR", 0)
            rssi_5g = state_dict.get(f"5G_{band}_RSSI", 0)
            
            snr_norm_5g = min(max(snr_5g / 30.0, 0), 1.0)
            rssi_norm_5g = min(max((rssi_5g + 100) / 40.0, 0), 1.0)
            
            qos_5g = (snr_norm_5g + rssi_norm_5g) / 2.0
            qos_reward += qos_5g * action_5g[i]
        
        # 3. Efficiency - higher data rates indicate better spectrum efficiency
        for i, band in enumerate(bands):
            datarate_4g = state_dict.get(f"4G_{band}_DataRate", 0)
            datarate_5g = state_dict.get(f"5G_{band}_DataRate", 0)
            
            # Normalize data rates (assuming reasonable upper bounds)
            datarate_norm_4g = min(datarate_4g / 100.0, 1.0)  # Adjust scaling as needed
            datarate_norm_5g = min(datarate_5g / 300.0, 1.0)  # 5G typically has higher data rates
            
            efficiency_reward += datarate_norm_4g * action_4g[i] + datarate_norm_5g * action_5g[i]
        
        # 4. Fairness - balance allocation across users
        for i, band in enumerate(bands):
            users_4g = state_dict.get(f"4G_{band}_Users", 0)
            users_5g = state_dict.get(f"5G_{band}_Users", 0)
            
            # More users should get proportionally more spectrum
            total_users = users_4g + users_5g
            if total_users > 0:
                fairness_4g = 1.0 - abs((users_4g / total_users) - action_4g[i])
                fairness_5g = 1.0 - abs((users_5g / total_users) - action_5g[i])
                fairness_reward += fairness_4g + fairness_5g
        
        # Combine rewards with appropriate weights
        # These weights can be adjusted based on importance
        w_throughput = 0.35
        w_qos = 0.35
        w_efficiency = 0.2
        w_fairness = 0.1
        
        # Scale down the total reward for better training stability
        total_reward = (w_throughput * throughput_reward +
                       w_qos * qos_reward +
                       w_efficiency * efficiency_reward +
                       w_fairness * fairness_reward) / 100.0
        
        # Add penalty for extreme allocations
        allocation_penalty = 0.1 * (np.std(action[:3]) + np.std(action[3:]))
        total_reward -= allocation_penalty
        
        return total_reward
        
            
    def reset(self, seed=None):
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)
        
        # Cycle to the next dataset for varied training
        self.current_dataset_idx = (self.current_dataset_idx + 1) % len(self.datasets)
        self.df = self.datasets[self.current_dataset_idx]
        self.max_steps = len(self.df)
        
        self.current_step = 0
        self.total_reward = 0
        self.rewards_history = []
        
        return self._get_state(), {}
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take a step in the environment with the given action."""
        # Calculate reward based on the action
        reward = self._calculate_reward(action)
        
        # Update tracking variables
        self.total_reward += reward
        self.rewards_history.append(reward)
        
        # Move to the next timestep
        self.current_step += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        # Get new state
        next_state = self._get_state()
        
        return next_state, reward, done, False, {}
    
    def render(self):
        """Render the environment (simplified)."""
        if self.current_step < len(self.df):
            print(f"Step: {self.current_step}, Time: {self.df.iloc[self.current_step].get('Time', 'N/A')}")
            print(f"Current reward: {self.rewards_history[-1] if self.rewards_history else 0}")
            print(f"Total reward: {self.total_reward}")
        else:
            print("Episode completed")

# Policy and Value networks for PPO
class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCritic, self).__init__()
        
        # Actor network (policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()  # Ensures positive outputs
        )
        
        # Critic network (value function)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        # Ensure state has correct shape
        if len(state.shape) == 1:
            state = state.unsqueeze(0)  # Add batch dimension
        
        # Get action distribution
        action_probs = self.actor(state)
        
        # Get state value
        value = self.critic(state)
        
        return action_probs, value
    
    def act(self, state):
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            # Convert state to float array first to handle mixed types
            state = np.array(state, dtype=np.float32)
            state = torch.FloatTensor(state).unsqueeze(0)
            
        # Get action allocation (without gradient calculation)
        with torch.no_grad():
            action_probs, _ = self.forward(state)
            
        # Convert to numpy array and ensure proper normalization
        action = action_probs.squeeze().numpy()
        
        # Ensure 4G and 5G allocations each sum to 1.0
        action_4g = action[0:3]
        action_5g = action[3:6]
        
        action_4g = action_4g / np.sum(action_4g) if np.sum(action_4g) > 0 else action_4g
        action_5g = action_5g / np.sum(action_5g) if np.sum(action_5g) > 0 else action_5g
        
        # Combine normalized allocations
        normalized_action = np.concatenate([action_4g, action_5g])
        
        return normalized_action
        
# PPO Agent
class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 lr: float = 1e-4, gamma: float = 0.99, eps_clip: float = 0.2, 
                 update_epochs: int = 10, entropy_coef: float = 0.01):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.update_epochs = update_epochs
        self.entropy_coef = entropy_coef
        
        # Initialize Actor-Critic network
        self.policy = ActorCritic(state_dim, action_dim, hidden_dim)
        
        # Initialize optimizer with learning rate scheduler
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)
        
        # Memory for storing trajectories
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.values = []
        self.is_terminals = []
        
    def select_action(self, state):
        # Get continuous allocation action
        return self.policy.act(state)
    
    def update(self):
        # Convert lists to tensors
        old_states = torch.FloatTensor(np.array(self.states, dtype=np.float32))
        old_actions = torch.FloatTensor(np.array(self.actions, dtype=np.float32))
        old_rewards = torch.FloatTensor(np.array(self.rewards, dtype=np.float32))
        old_is_terminals = np.array(self.is_terminals)
        
        # Calculate returns and normalize
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(old_rewards), reversed(old_is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        returns = torch.FloatTensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Update policy for multiple epochs
        for _ in range(self.update_epochs):
            # Get current action probabilities and state values
            action_probs, state_values = self.policy(old_states)
            state_values = state_values.squeeze()
            
            # Calculate advantages
            advantages = returns - state_values.detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Calculate losses
            # Actor loss with entropy regularization
            entropy = -torch.mean(torch.sum(action_probs * torch.log(action_probs + 1e-10), dim=1))
            actor_loss = -torch.mean(advantages.unsqueeze(1) * torch.log(action_probs + 1e-10))
            actor_loss = actor_loss - self.entropy_coef * entropy
            
            # Critic loss with L1 smoothing
            critic_loss = torch.mean((returns - state_values) ** 2) + \
                         0.1 * torch.mean(torch.abs(returns - state_values))
            
            # Total loss
            loss = actor_loss + 0.5 * critic_loss
            
            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
            self.optimizer.step()
        
        # Step the learning rate scheduler
        self.scheduler.step()
        
        # Clear memory
        self.states = []
        self.actions = []
        self.rewards = []
        self.is_terminals = []
        
# Training function
def train(env, agent, num_episodes: int = 1000, eval_interval: int = 25, 
          early_stop_threshold: float = 0.85):
    """Train the PPO agent with improved monitoring and early stopping."""
    rewards = []
    best_reward = float('-inf')
    patience = 10
    patience_counter = 0
    
    # Training loop
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            action = agent.select_action(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Store in memory
            agent.states.append(state)
            agent.actions.append(action)
            agent.rewards.append(reward)
            agent.is_terminals.append(done)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        # Update agent after episode
        agent.update()
        
        # Track rewards
        rewards.append(episode_reward)
        
        # Evaluate and save best model
        if (episode + 1) % eval_interval == 0:
            avg_reward = np.mean(rewards[-eval_interval:])
            print(f"Episode {episode+1}/{num_episodes}, Avg Reward: {avg_reward:.3f}")
            
            if avg_reward > best_reward:
                best_reward = avg_reward
                model_path = os.path.join(os.path.dirname(__file__), 'best_spectrum_allocation_model.pth')
                torch.save(agent.policy.state_dict(), model_path)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if avg_reward >= early_stop_threshold or patience_counter >= patience:
                print(f"Early stopping at episode {episode+1}")
                break
    
    return rewards

# Evaluation function
def evaluate(env, agent, num_episodes: int = 10):
    """Evaluate the trained agent performance."""
    rewards = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        actions_history = []
        states_history = []
        
        while not done:
            # Select action without exploration
            action = agent.select_action(state)
            actions_history.append(action)
            states_history.append(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
        
        rewards.append(episode_reward)
        
        # Print episode results
        print(f"Evaluation Episode {episode+1}/{num_episodes}, Total Reward: {episode_reward:.3f}")
        
    return np.mean(rewards), actions_history, states_history

# Main function
def main():
    """Main function to set up and run the PPO training for spectrum allocation."""
    # Initialize environment
    env = SpectrumAllocationEnv(data_dir=os.path.join(os.path.dirname(__file__), "spectrum_demand_data"), file_pattern="spectrum_demand_*.csv")
    
    # Get state and action dimensions
    state_dim = len(env._get_features())
    action_dim = env.action_space.shape[0]
    
    # Initialize agent with modified parameters
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        eps_clip=0.2,
        update_epochs=5,
        entropy_coef=0.02
    )
    
    # Train agent with modified parameters
    rewards = train(
        env, 
        agent, 
        num_episodes=1000,
        eval_interval=25,
        early_stop_threshold=0.85
    )
    
    # Evaluate agent
    avg_reward, actions, states = evaluate(env, agent)
    
    # Save trained model
    model_path = os.path.join(os.path.dirname(__file__), 'spectrum_allocation_model.pth')
    torch.save(agent.policy.state_dict(), model_path)
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.title('Training Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    figure_path = os.path.join(os.path.dirname(__file__), 'training_rewards.png')
    plt.savefig(figure_path)
    plt.show()

if __name__ == "__main__":
    main()

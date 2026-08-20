import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from spectrum_allocation_ppo import SpectrumAllocationEnv, ActorCritic, PPOAgent

def test_trained_model(model_path=os.path.join(os.path.dirname(__file__), 'spectrum_allocation_model.pth'), num_episodes=10, visualize=True):
    """
    Test a trained spectrum allocation model.
    
    Args:
        model_path: Path to the saved model
        num_episodes: Number of episodes to test
        visualize: Whether to visualize the results
    
    Returns:
        Average reward across test episodes
    """
    # Initialize environment
    env = SpectrumAllocationEnv(data_dir="spectrum_demand_data", file_pattern="spectrum_demand_*.csv")
    
    # Get state and action dimensions
    state_dim = len(env._get_features())
    action_dim = env.action_space.shape[0]
    
    # Initialize agent
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim)
    
    # Load trained model
    if os.path.exists(model_path):
        agent.policy.load_state_dict(torch.load(model_path))
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file {model_path} not found!")
    
    # Test the model
    rewards = []
    all_actions = []
    all_states = []
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        episode_actions = []
        episode_states = []
        
        while not done:
            # Select action without exploration
            action = agent.select_action(state)
            episode_actions.append(action)
            episode_states.append(state)
            
            # Take action
            next_state, reward, done, _, _ = env.step(action)
            
            # Update state and reward
            state = next_state
            episode_reward += reward
            
            # Optionally render the environment
            if visualize:
                env.render()
        
        rewards.append(episode_reward)
        all_actions.append(episode_actions)
        all_states.append(episode_states)
        
        # Print episode results
        print(f"Test Episode {episode+1}/{num_episodes}, Total Reward: {episode_reward:.3f}")
    
    avg_reward = np.mean(rewards)
    print(f"Average Reward over {num_episodes} episodes: {avg_reward:.3f}")
    
    if visualize:
        # Plot rewards
        plt.figure(figsize=(10, 5))
        plt.plot(rewards)
        plt.title('Test Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.savefig(os.path.join(os.path.dirname(__file__), 'test_rewards.png'))
        plt.show()
        
        # Plot allocation strategies for the last episode
        if all_actions:
            last_episode_actions = np.array(all_actions[-1])
            
            plt.figure(figsize=(12, 8))
            
            # Plot 4G allocations
            plt.subplot(2, 1, 1)
            plt.plot(last_episode_actions[:, 0], label='Low Band')
            plt.plot(last_episode_actions[:, 1], label='Mid Band')
            plt.plot(last_episode_actions[:, 2], label='High Band')
            plt.title('4G Spectrum Allocation')
            plt.xlabel('Time Step')
            plt.ylabel('Allocation Proportion')
            plt.legend()
            plt.grid(True)
            
            # Plot 5G allocations
            plt.subplot(2, 1, 2)
            plt.plot(last_episode_actions[:, 3], label='Low Band')
            plt.plot(last_episode_actions[:, 4], label='Mid Band')
            plt.plot(last_episode_actions[:, 5], label='High Band')
            plt.title('5G Spectrum Allocation')
            plt.xlabel('Time Step')
            plt.ylabel('Allocation Proportion')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.dirname(__file__), 'allocation_strategy.png'))
            plt.show()
    
    return avg_reward, all_actions, all_states

def analyze_allocation_patterns(actions, states, env):
    """Analyze the allocation patterns in relation to the environment state."""
    if not actions or not states:
        print("No data to analyze")
        return
    
    # Analyze the last episode
    episode_actions = np.array(actions[-1])
    episode_states = np.array(states[-1])
    features = env._get_features()
    
    # Create time-based analysis plots
    plt.figure(figsize=(15, 10))
    
    # Plot 4G allocations vs demand over time
    plt.subplot(2, 1, 1)
    bands = ["Low", "Mid", "High"]
    for i, band in enumerate(bands):
        # Get demand index
        demand_idx = next((j for j, f in enumerate(features) if f == f"4G_{band}_Demand"), None)
        if demand_idx is not None:
            demand = episode_states[:, demand_idx]
            # Scale allocation to match demand scale (multiply by 100)
            allocation = episode_actions[:, i] * 100
            
            # Plot both demand and allocation
            plt.plot(range(24), demand[:24], '--', label=f'{band} Band Demand', alpha=0.6)
            plt.plot(range(24), allocation[:24], '-', label=f'{band} Band Allocation')
    
    plt.title('4G Demand vs Allocation over 24 Hours')
    plt.xlabel('Hour')
    plt.ylabel('Demand / Allocation (%)')
    plt.legend()
    plt.grid(True)
    
    # Plot 5G allocations vs demand over time
    plt.subplot(2, 1, 2)
    for i, band in enumerate(bands):
        # Get demand index
        demand_idx = next((j for j, f in enumerate(features) if f == f"5G_{band}_Demand"), None)
        if demand_idx is not None:
            demand = episode_states[:, demand_idx]
            # Scale allocation to match demand scale (multiply by 100)
            allocation = episode_actions[:, i+3] * 100
            
            # Plot both demand and allocation
            plt.plot(range(24), demand[:24], '--', label=f'{band} Band Demand', alpha=0.6)
            plt.plot(range(24), allocation[:24], '-', label=f'{band} Band Allocation')
    
    plt.title('5G Demand vs Allocation over 24 Hours')
    plt.xlabel('Hour')
    plt.ylabel('Demand / Allocation (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'time_based_allocation.png'))
    plt.show()
    
    # Calculate and print hourly statistics
    print("\nHourly Allocation Analysis:")
    for hour in range(24):
        print(f"\nHour {hour}:")
        print("4G Allocations:")
        for i, band in enumerate(bands):
            demand_idx = next((j for j, f in enumerate(features) if f == f"4G_{band}_Demand"), None)
            if demand_idx is not None:
                demand = episode_states[hour, demand_idx]
                # Scale allocation to percentage
                allocation = episode_actions[hour, i] * 100
                print(f"  {band} Band - Demand: {demand:.2f}%, Allocation: {allocation:.2f}%")
        
        print("5G Allocations:")
        for i, band in enumerate(bands):
            demand_idx = next((j for j, f in enumerate(features) if f == f"5G_{band}_Demand"), None)
            if demand_idx is not None:
                demand = episode_states[hour, demand_idx]
                # Scale allocation to percentage
                allocation = episode_actions[hour, i+3] * 100
                print(f"  {band} Band - Demand: {demand:.2f}%, Allocation: {allocation:.2f}%")

def main():
    """Main function to test the trained model."""
    # Test the trained model
    model_path = os.path.join(os.path.dirname(__file__), 'spectrum_allocation_model.pth')
    env = SpectrumAllocationEnv(data_dir=os.path.join(os.path.dirname(__file__), "spectrum_demand_data"), file_pattern="spectrum_demand_*.csv")
    
    try:
        avg_reward, actions, states = test_trained_model(model_path=model_path, num_episodes=5, visualize=True)
        
        # Analyze allocation patterns
        analyze_allocation_patterns(actions, states, env)
        
        print(f"\nTesting completed successfully with average reward: {avg_reward:.3f}")
    except Exception as e:
        print(f"Error during testing: {e}")

if __name__ == "__main__":
    main()
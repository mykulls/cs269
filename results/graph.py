import numpy as np
import matplotlib.pyplot as plt

# Initialize empty lists to store ep_rew_mean and total_timesteps
ep_rew_mean = []
total_timesteps = []

# Open the file and read the lines
with open('ppo_rewards.txt', 'r') as file:
    for line in file:
        # Split the line by '|' and strip extra spaces
        parts = [part.strip() for part in line.split('|')]
        
        if len(parts) >= 3:  # Ensure the line has the expected format
            key, value = parts[1], parts[2]
            # Check if the line is for ep_rew_mean or total_timesteps
            if 'ep_rew_mean' in key:
                ep_rew_mean.append(float(value))
            elif 'total_timesteps' in key:
                total_timesteps.append(int(value))

# Convert the lists to numpy arrays
ep_rew_mean_array = np.array(ep_rew_mean)
total_timesteps_array = np.array(total_timesteps)

# Print the resulting arrays
print("ep_rew_mean array:", ep_rew_mean_array)
print("total_timesteps array:", total_timesteps_array)

plt.figure(figsize=(10, 6))
plt.plot(total_timesteps_array, ep_rew_mean_array, marker='o', linestyle='-', color='b', label='Ep Rew Mean')

# Add labels, title, and legend
plt.xlabel('Total Timesteps', fontsize=14)
plt.ylabel('Episode Reward Mean', fontsize=14)
plt.title('Episode Reward Mean vs Total Timesteps (PPO Algorithm)', fontsize=16)
plt.legend(fontsize=12)

# Show grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()

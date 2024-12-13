import numpy as np
import matplotlib.pyplot as plt

# Initialize empty lists to store ep_rew_mean and total_timesteps
reward = []
# total_timesteps = []

# Open the file and read the lines
with open('unscaled_lift_reward.txt', 'r') as file:
    for line in file:
        # Split the line by '|' and strip extra spaces
        parts = [part.strip() for part in line.split(':')]
        key, value = parts[0], parts[1]
        # Check if the line is for ep_rew_mean or total_timesteps
        if 'Reward' in key:
            reward.append(float(value))

# Convert the lists to numpy arrays
reward_arr = np.array(reward)

# Print the resulting arrays
print("reward array:", reward_arr)

plt.figure(figsize=(10, 6))
plt.plot(reward_arr, marker='o', linestyle='-', color='b', label='Reward')

# Add labels, title, and legend
plt.xlabel('Timesteps', fontsize=14)
plt.ylabel('Un-normalized Reward', fontsize=14)
plt.title('Reward vs Timesteps During An Episode (Hard-coded Lift)', fontsize=16)
plt.legend(fontsize=12)

# Show grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Display the plot
plt.show()

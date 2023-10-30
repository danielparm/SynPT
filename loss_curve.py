import re
import matplotlib.pyplot as plt
import numpy as np

# Initialize empty lists to store the extracted values
import re
import numpy as np
import matplotlib.pyplot as plt

# Initialize empty lists to store the extracted values
steps = []
train_losses = []
val_losses = []
train_times = []

# Regular expression patterns to match and extract the desired values
step_pattern = r'step ([\d.]+):'
train_loss_pattern = r'train loss ([\d.]+),'
val_loss_pattern = r'val loss ([\d.]+) '
train_time_pattern = r'training time ([\d.]+) '

# Read the contents of the Nohup.out file
with open('nohup.out', 'r') as f:
    data = f.read()

# Using regular expressions to extract the desired values
for line in data.split('\n'):
    step_match = re.search(step_pattern, line)
    train_loss_match = re.search(train_loss_pattern, line)
    val_loss_match = re.search(val_loss_pattern, line)
    train_time_match = re.search(train_time_pattern, line)

    if step_match and train_loss_match and val_loss_match and train_time_match:
        step = float(step_match.group(1))
        train_loss = float(train_loss_match.group(1))
        val_loss = float(val_loss_match.group(1))
        train_time = float(train_time_match.group(1))

        steps.append(step)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_times.append(train_time)

window_size = 5
train_losses_ma = np.convolve(train_losses[5:], np.ones(window_size) / window_size, mode='valid')
val_losses_ma = np.convolve(val_losses[5:], np.ones(window_size) / window_size, mode='valid')

# Plotting the moving average loss curve against training time
plt.plot(train_times[window_size - 1 + 5:], train_losses_ma, label='Moving Average Train Loss')
plt.plot(train_times[window_size - 1 + 5:], val_losses_ma, label='Moving Average Validation Loss')
plt.xlabel('Training Time (minutes)')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'loss_curve{train_times[-1]}.png')
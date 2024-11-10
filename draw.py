import matplotlib.pyplot as plt
import numpy as np

groups = ['100', '200']
batch_sz128 = [59.81962203979492, 63.069854736328125]
batch_sz256 = [61.03630447387695, 64.12914276123047]
batch_sz512 = [62.318477630615234, 65.48713684082031]
# Number of groups
n_groups = len(groups)

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Set the bar width
bar_width = 0.08

# Set the opacity
opacity = 1.0

# Set the positions of the bars on the x-axis
index = np.arange(n_groups)

# Plot the bars
rects1 = plt.bar(index, batch_sz128, bar_width, alpha=opacity, color='b', label='128')
rects2 = plt.bar(index + bar_width, batch_sz256, bar_width, alpha=opacity, color='orange', label='256')
rects2 = plt.bar(index + 2*bar_width, batch_sz512, bar_width, alpha=opacity, color='g', label='512')

# Add labels, title, and legend
plt.xlabel('Training epoches')
plt.ylabel('Top-1')
plt.title('Top-1 Accuracy by Group and Epochs')
plt.xlim(-1, 2)
plt.xticks(index + bar_width, groups)
plt.ylim(50, 70)
plt.yticks(np.arange(50, 71, 2.5))

plt.legend(title='Batch size')

# Display the chart
plt.tight_layout()
plt.show()
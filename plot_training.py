import matplotlib.pyplot as plt
iterations = []
losses = []
accuracies = []
with open('result.txt', 'r') as file:
    for line in file:
        if "Iteration :" in line:
            parts = line.split('|')
            it = int(parts[0].split(':')[1].strip())
            loss = float(parts[1].split(':')[1].strip())
            acc = float(parts[2].split(':')[1].strip())
            iterations.append(it)
            losses.append(loss)
            accuracies.append(acc)
fig, ax1 = plt.subplots(figsize=(10, 6))
color = 'tab:red'
ax1.set_xlabel('Iterations (Epochs)', fontweight='bold')
ax1.set_ylabel('Categorical Cross-Entropy Loss', color=color, fontweight='bold')
ax1.plot(iterations, losses, color=color, linewidth=2, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)
ax2 = ax1.twinx()
color = 'tab:blue'
ax2.set_ylabel('Accuracy (%)', color=color, fontweight='bold')
ax2.plot(iterations, accuracies, color=color, linewidth=2, label='Accuracy')
ax2.tick_params(axis='y', labelcolor=color)
plt.title('Neural Network Training Progress (50,000 Iterations)', fontsize=14, fontweight='bold')
fig.tight_layout() 
plt.show()
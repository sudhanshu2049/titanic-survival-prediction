import matplotlib.pyplot as plt
import numpy as np

# Simulate training history data
epochs = np.arange(1, 11)
train_acc = np.array([0.68, 0.75, 0.80, 0.81, 0.83, 0.84, 0.85, 0.85, 0.86, 0.86])
val_acc = np.array([0.65, 0.72, 0.76, 0.77, 0.78, 0.78, 0.79, 0.79, 0.80, 0.80])
train_loss = np.array([0.62, 0.52, 0.45, 0.44, 0.40, 0.38, 0.36, 0.36, 0.35, 0.35])
val_loss = np.array([0.65, 0.55, 0.50, 0.50, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54])

# Plot 1: Initial training
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs, train_acc, label='Train', marker='o')
ax1.plot(epochs, val_acc, label='Validation', marker='o')
ax1.set_title('Model accuracy', fontsize=14)
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, train_loss, label='Train', marker='o')
ax2.plot(epochs, val_loss, label='Validation', marker='o')
ax2.set_title('Model loss', fontsize=14)
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/accuracy_loss.png', dpi=150, bbox_inches='tight')
print('Saved results/accuracy_loss.png')
plt.close()

# Plot 2: Tuned model (30 epochs)
epochs_tuned = np.arange(1, 31)
train_acc_tuned = np.linspace(0.76, 0.85, 30) + np.random.normal(0, 0.01, 30)
val_acc_tuned = np.linspace(0.74, 0.81, 30) + np.random.normal(0, 0.01, 30)
train_loss_tuned = np.linspace(0.62, 0.38, 30) + np.random.normal(0, 0.02, 30)
val_loss_tuned = np.linspace(0.65, 0.50, 30) + np.random.normal(0, 0.02, 30)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.plot(epochs_tuned, train_acc_tuned, label='Train', color='C0')
ax1.plot(epochs_tuned, val_acc_tuned, label='Validation', color='C1')
ax1.set_title('Model Accuracy - Best Tuned Model', fontsize=14)
ax1.set_ylabel('Accuracy')
ax1.set_xlabel('Epoch')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_tuned, train_loss_tuned, label='Train', color='C0')
ax2.plot(epochs_tuned, val_loss_tuned, label='Validation', color='C1')
ax2.set_title('Model Loss - Best Tuned Model', fontsize=14)
ax2.set_ylabel('Loss')
ax2.set_xlabel('Epoch')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/accuracy_loss_tuned.png', dpi=150, bbox_inches='tight')
print('Saved results/accuracy_loss_tuned.png')
plt.close()

import matplotlib.pyplot as plt

epochs = list(range(1, 33))
train_loss = [
    0.9747, 0.9129, 0.9048, 0.9006, 0.8978, 0.8958, 0.8943,
    0.8931, 0.8920, 0.8912, 0.8905, 0.8898, 0.8893, 0.8888,
    0.8883, 0.8880, 0.8876, 0.8872, 0.8869, 0.8867, 0.8864,
    0.8861, 0.8859, 0.8857, 0.8854, 0.8852, 0.8850, 0.8849,
    0.8847, 0.8845, 0.8844, 0.8843
]

plt.figure(figsize=(8,5))
plt.plot(epochs, train_loss, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Avg Train Loss (MSE)')
plt.title('Training Loss Curve')
plt.grid(True)
plt.tight_layout()
plt.show()
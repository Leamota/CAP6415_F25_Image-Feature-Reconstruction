import torch
import os

# Use your actual variable names: decoder (model), optimizer, last epoch, last loss
save_dir = "/content/checkpoints"
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "decoder_final.pth")

torch.save({
    'epoch': epochs,
    'model_state_dict': decoder.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': float(loss.item()),  # Or avg_loss if you store it
}, save_path)

print(f"Model saved to {save_path}")
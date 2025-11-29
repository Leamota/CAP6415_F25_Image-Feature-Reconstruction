import os

os.makedirs("results", exist_ok=True)
num_images = min(images.shape[0], 8)

for i in range(num_images):
    # Original image (denormalized for viewing)
    orig = images[i].detach().cpu().permute(1,2,0).numpy()
    orig = (orig - orig.min()) / (orig.max() - orig.min())
    plt.imsave(f"results/original_{i}.png", orig)

    # Reconstructed image
    recon_img = recon[i].detach().cpu().permute(1,2,0).numpy()
    plt.imsave(f"results/reconstruction_{i}.png", recon_img)

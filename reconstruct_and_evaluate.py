decoder.eval()
with torch.no_grad():
    feat_cat = extract_and_concat_features(images, extractor)
    recon = decoder(feat_cat)

plt.figure(figsize=(16, 4))
for i in range(min(images.shape[0], 8)):
    plt.subplot(2, 8, i+1)
    img = images[i].detach().cpu().permute(1,2,0).numpy()
    img = (img - img.min())/(img.max() - img.min())
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')
    plt.subplot(2, 8, i+9)
    rimg = recon[i].detach().cpu().permute(1,2,0).numpy()
    plt.imshow(rimg)
    plt.title("Recon")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Metrics
orig = images.cpu().detach().numpy()
reco = recon.cpu().detach().numpy()
mse = np.mean((orig - reco) ** 2)
psnr = np.mean([peak_signal_noise_ratio(o, r, data_range=1.0) for o, r in zip(orig, reco)])
ssim = np.mean([structural_similarity(np.transpose(o, (1,2,0)), np.transpose(r, (1,2,0)), win_size=11, data_range=1.0, channel_axis=2) for o, r in zip(orig, reco)])
print(f"MSE: {mse:.4f} | PSNR: {psnr:.2f} | SSIM: {ssim:.4f}")

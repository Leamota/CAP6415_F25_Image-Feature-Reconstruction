import torch.optim as optim

decoder.train()
optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

epochs = 32
for epoch in range(epochs):
    running_loss = 0.0
    for images, _ in loader:
        images = images.to(device)
        feat_cat = extract_and_concat_features(images, extractor)
        recon = decoder(feat_cat)
        loss = loss_fn(recon, images)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    avg_loss = running_loss / len(loader.dataset)
    print(f"Epoch {epoch+1}/{epochs} - Avg Train Loss: {avg_loss:.4f}")

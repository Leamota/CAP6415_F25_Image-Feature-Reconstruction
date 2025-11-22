def extract_and_concat_features(images, extractor):
    with torch.no_grad():
        feats = extractor(images)
        spatial_size = feats[-1].shape[2:]
        up_feats = [F.interpolate(f, size=spatial_size, mode='bilinear') for f in feats]
        feat_cat = torch.cat(up_feats, dim=1)
    return feat_cat  # (batch, total_channels, H, W)

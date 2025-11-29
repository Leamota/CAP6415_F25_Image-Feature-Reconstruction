# Get a batch and example features
images, labels = next(iter(loader))
images = images.to(device)
feat_cat = extract_and_concat_features(images, extractor)
in_ch = feat_cat.shape[1]

decoder = ConvOnlyDecoder(in_ch).to(device)
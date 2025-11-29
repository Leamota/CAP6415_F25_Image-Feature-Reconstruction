# Path to class-organized validation folder
data_dir = '/content/ILSVRC2012_val_foldered'

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

dataset = datasets.ImageFolder(root=data_dir, transform=preprocess)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
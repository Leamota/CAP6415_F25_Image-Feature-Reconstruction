device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torchvision.models import ResNet50_Weights

resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1).to(device)
resnet50.eval()

# Define hook layers (from early, mid, deep, deepest)
layer_names = ["layer1.2.relu", "layer2.3.relu", "layer3.5.relu", "layer4.2.relu"]

class FeatureExtractor(nn.Module):
    def __init__(self, model, layer_names):
        super().__init__()
        self.model = model
        self.layers = layer_names
        self._outputs = {}
        self._register_hooks()
    def _register_hooks(self):
        for name in self.layers:
            layer = dict(self.model.named_modules())[name]
            layer.register_forward_hook(self._make_hook(name))
    def _make_hook(self, name):
        def hook(module, input, output):
            self._outputs[name] = output
        return hook
    def forward(self, x):
        self._outputs = {}
        _ = self.model(x)
        # Output list in specified layer order
        return [self._outputs[name] for name in self.layers]

extractor = FeatureExtractor(resnet50, layer_names)

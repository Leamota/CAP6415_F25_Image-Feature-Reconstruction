class ConvOnlyDecoder(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 512, 4, stride=2, padding=1),   # 7x7 -> 14x14
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),     # 14x14 -> 28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),     # 28x28 -> 56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),      # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),        # 112x112 -> 224x224
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.up(x)

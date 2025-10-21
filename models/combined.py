from decoder import MyLSTM
from encoder import MyVGG16

class CombinedModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs, _ = self.decoder(captions, None)
        return outputs

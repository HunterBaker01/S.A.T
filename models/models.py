from torch import nn
from torchvision.models import vgg16
from torch.nn import LSTM
from torch import cat

class MyVGG16(nn.Module):
    """
    :param freeze_features: use the pretrained vgg16 or not
    """
    def __init__(self, freeze_features=True):
        super().__init__()
        model = vgg16(weights="DEFAULT")

        self.feature= nn.Sequential(*list(model.features.children()))

        if freeze_features:
            for param in self.feature.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.feature(x) # [batch_size, 512, 7, 7]
        return x

class MyLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        """
        :param embedding_dim: embedding size
        :param hidden_dim: hidden size
        :param vocab_size: size of vocaublary
        :param num_layers: number of layers
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim) # [embedding_dim, input_dim] 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True) # Outputs [N, L, D * H_out]
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        captions_in = captions[:, :-1] # Don't let the model see the answer
        emb = self.embedding(captions_in)
        features = features.unsqueeze(1)
        lstm_input = cat((features, emb), dim=1) # Input both the output of the cnn and the embedded caption
        outputs, _ = self.lstm(lstm_input)
        outputs = self.fc(outputs)
        return outputs

class CombinedModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

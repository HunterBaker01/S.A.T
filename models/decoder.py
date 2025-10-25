from torch.nn import LSTM
from torch import nn

class MyLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        captions_in = captions[:, :-1]
        emb = self.embedding(captions_in)
        features = features.unsqueeze(1)
        lstm_input = torch.cat((features, emb), dim=1)
        outputs, _ = self.lstm(lstm_input)
        outputs = self.fc(outputs)
        return outputs

from torch.nn import LSTM
from torch import nn

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, embedding_dim, num_layers=1):
        super(MyLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, captions):
        batch_size = x.size(0)
        x = x.view(batch_size, 512, 14*14).permute(0, 2, 1) # [batch, 196, 512]
        img_features = x.mean(dim=1) # [batch, 512]
        img_features = img_features.unsqueeze(1) # [batch, 1, 512]

        embeddings = self.embedding(captions) # [batch, seq_len, embedding_dim]

        combined = torch.cat((img_feature, embeddings), dim=1)  # [batch, seq_len+1, 512]

        out, _ = self.lstm(combined)
        out = self.fc(out) # [batch, 196, vocab_size]
        return out

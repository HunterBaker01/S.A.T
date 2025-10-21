from torch.nn import LSTM
from torch import nn

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers=1):
        super(MyLSTM, self).__init__()
        self.lstm = LSTM(input_size, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 512, 14*14).permute(0, 2, 1) # [batch, 196, 512]
        out, hidden = self.lstm(x)
        out = self.fc(out) # [batch, 196, vocab_size]
        return out

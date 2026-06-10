import torch
import torch.nn as nn

class LSTMembedding(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=30, num_layers=1, bidirectional=True):
        super(LSTMembedding, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, bidirectional=bidirectional, batch_first=True)
        
        fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        lstm_out, _ = self.lstm(x)
        
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        return self.fc(last_hidden)
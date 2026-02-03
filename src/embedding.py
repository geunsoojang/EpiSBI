import torch
import torch.nn as nn

class LSTMembedding(nn.Module):
    """
    LSTM-based embedding network for time-series data.
    Used in NPE-LSTM and PNPE to compress trajectories into a latent feature vector.
    """
    def __init__(self, input_dim=1, hidden_dim=128, output_dim=30, num_layers=1, bidirectional=True):
        super(LSTMembedding, self).__init__()
        
        # Define the LSTM layer
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            bidirectional=bidirectional, 
            batch_first=True
        )
        
        # Calculate the size of the features coming out of LSTM
        fc_input_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Fully connected layer to map LSTM output to the desired summary feature size
        self.fc = nn.Linear(fc_input_dim, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Ensure input is 3D: (batch, seq_len, input_dim)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        # lstm_out shape: (batch, seq_len, hidden_dim * num_directions)
        lstm_out, _ = self.lstm(x)
        
        # Extract the hidden state of the last time step
        last_hidden = lstm_out[:, -1, :]
        last_hidden = self.dropout(last_hidden)
        
        # Output the summary features
        return self.fc(last_hidden)
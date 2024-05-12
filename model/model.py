import torch
from torch import nn

class LSTMClassifier(nn.Module):
    """this is LSTM custom model class"""
    def __init__(self):
        super(LSTMClassifier, self).__init__()
        self.__hidden_size = 128
        self.__input_size = 128
        self.__num_layers = 2
        self.__num_classes = 5
        self.__dropout = 0.2
        self.lstm = nn.LSTM(self.__input_size, self.__hidden_size, self.__num_layers, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(self.__hidden_size, self.__hidden_size//2),
            nn.ReLU(),
            nn.Dropout(self.__dropout),

            nn.Linear(self.__hidden_size//2, self.__hidden_size//4),
            nn.ReLU(),
            nn.Dropout(self.__dropout),

            nn.Linear(self.__hidden_size//4, self.__num_classes),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))

        # Flatten the output from LSTM layer if needed
        out = out[:, -1, :]  # Get the last time step output
        # You may need to flatten the output if needed, depending on the shape

        out = self.fc(out)
        return out
    

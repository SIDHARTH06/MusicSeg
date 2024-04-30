import torch
import torch
import torch.nn as nn
import torch.optim as optim

class MusicLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(MusicLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Define LSTM units for each window
        self.lstm_layers = nn.ModuleList([nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)
                                           for _ in range(output_size)])

        # Define linear layers to map LSTM output to single sigmoid output
        self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(output_size)])

    def forward(self, x):
        outputs = []
        for i in range(x.size(1)):
            lstm_out, _ = self.lstm_layers[i](x[:, i:i+1, :])
            linear_out = self.linear_layers[i](lstm_out[:, -1, :])  # Apply linear layer to LSTM output
            sigmoid_out = torch.sigmoid(linear_out)  # Apply sigmoid activation
            outputs.append(sigmoid_out)

        return torch.cat(outputs, dim=1)
    

class ThreeTierEnsemble():
    def __init__(self,model30path,model10path,model1path):
        self.model30 = torch.load(model30path)
        self.model10 = torch.load(model10path)
        self.model1 = torch.load(model1path)
    def predict(self,x):
        out30 = self.model30(x).detach().cpu().numpy()[0] > 0.5
        # for i,x in enumerate(x):


# -------TODO--------------


import torch
import torch
import torch.nn as nn
import torch.optim as optim
THRESHOLD =0.5
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
    
import librosa
import numpy as np
class ThreeTierEnsemble():
    def __init__(self,model30path,model10path,model1path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model30 = torch.load(model30path)
        self.model10 = torch.load(model10path)
        self.model1 = torch.load(model1path)
        self.model30.to(self.device)
        self.model10.to(self.device)
        self.model1.to(self.device)
    def extract_mfcc(self,y, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, duration=5):
        num_segments = int(len(y) / (duration * sr))
        mfccs = []
        for i in range(num_segments):
            segment = y[i * duration * sr : (i + 1) * duration * sr]
            mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
            mfcc_mean = np.mean(mfcc, axis=1)
            mfccs.append(mfcc_mean)

        return np.stack(mfccs, axis=0)
    def predict(self,y):
        x = self.extract_mfcc(y,duration=30)
        out30 = list(self.model30(torch.tensor(x,dtype=torch.float).to(self.device).detach().cpu().numpy()[0] >= THRESHOLD))
        nextinp = []
        nx = self.extract_mfcc(y,duration=10)
        for i,data in enumerate(x):
            if out30[i]:
                nextinp.append(nx[(i*3):(i*3)+3])
            else:
                nextinp.append([[0]*13]*3)
        nextinp=torch.tensor(np.array(nextinp))
        #-----TODO-----



                





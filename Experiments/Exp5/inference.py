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
    def pad_array(self,array):
        padded_array = np.pad(array, ((0, 13 - array.shape[0]), (0, 13 - array.shape[1])), mode='constant', constant_values=0)
        return padded_array
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
        x= self.pad_array(x)
        x=np.reshape(x,(1,x.shape[0],x.shape[1]))
        out30 = self.model30(torch.tensor(x,dtype=torch.float).to(self.device)).detach().cpu().numpy()[0] >= THRESHOLD
        nextinp = np.zeros((41,13))
        zeropad=np.zeros(13)
        nx = self.extract_mfcc(y,duration=10)
        for i,data in enumerate(x[0]):
            if out30[i]:
                nextinp[i*3]=nx[i*3]
                nextinp[(i*3)+1]=nx[(i*3)+1]
                nextinp[(i*3)+2]=nx[(i*3)+2]
            else:
                nextinp[i*3]=zeropad
                nextinp[(i*3)+1]=zeropad
                nextinp[(i*3)+2]=zeropad
        nextinp=torch.tensor(nextinp,dtype=torch.float).to(self.device)

        # 10 second model
        x=np.reshape(nextinp,(1,nextinp.shape[0],nextinp.shape[1]))
        print(x.shape)
        out10 = self.model10(torch.tensor(x,dtype=torch.float).to(self.device)).detach().cpu().numpy()[0] >= THRESHOLD
        nextinp = np.zeros((416,13))
        zeropad=np.zeros(13)
        nx = self.extract_mfcc(y,duration=1)
        for i,data in enumerate(x[0]):
            if out10[i]:
                nextinp[i*3]=nx[i*3]
                nextinp[(i*3)+1]=nx[(i*3)+1]
                nextinp[(i*3)+2]=nx[(i*3)+2]
            else:
                nextinp[i*3]=zeropad
                nextinp[(i*3)+1]=zeropad
                nextinp[(i*3)+2]=zeropad
        nextinp=torch.tensor(nextinp,dtype=torch.float).to(self.device)

        #1 second model

        x=np.reshape(nextinp,(1,nextinp.shape[0],nextinp.shape[1]))
        out1 = self.model1(torch.tensor(x,dtype=torch.float).to(self.device)).detach().cpu().numpy()[0] >= THRESHOLD
        return out1



model = ThreeTierEnsemble(model10path='/Users/sidharthdeepesh/MusicSeg/Experiments/Exp5/Models/model10.pt',model1path='/Users/sidharthdeepesh/MusicSeg/Experiments/Exp5/Models/model1.pt',model30path='/Users/sidharthdeepesh/MusicSeg/Experiments/Exp5/Models/model30.pt')
y,_ = librosa.load('/Users/sidharthdeepesh/MusicSeg/Dataset A/Raw/Songs/004.mp3', sr=22050)
out = model.predict(y) > THRESHOLD
true_indices = [index for index, value in enumerate(out) if value]
print(len(y)//22050)
print("Indices:", (np.array(true_indices)/86)*324)



                





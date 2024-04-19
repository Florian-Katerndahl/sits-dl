"""
LSTM implementation adapted from https://github.com/JKfuberlin/SITST4TSC
"""
import torch
from torch import nn, Tensor
from typing import Optional
import numpy as np
from sits_dl.tensordatacube import TensorDataCube as TDC

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LSTMClassifier(nn.Module):
    def __init__(self, num_bands: int, input_size: int, hidden_size: int, num_layers: int, num_classes: int,
                 bidirectional: bool):
        super(LSTMClassifier, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 1
        if bidirectional:
            self.D = 2
        self.embd = nn.Linear(num_bands, input_size)
        # definitions of forget gate, memory etc are defined within nn.LSTM:
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.D * hidden_size, 256),  # hidden layer 1, amount of neurons in input layer get reduced to 256
            nn.ReLU(),  # activation function
            nn.BatchNorm1d(256),  # mal rausnehmen
            nn.Dropout(0.3),  # dropping units at random to prevent overfitting
            nn.Linear(256, num_classes),  # hidden layer 2
            nn.Softmax(dim=1)  # final layer to calculate probablitiy for each class
        )

    def forward(self, x: Tensor):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = self.embd(x)
        h0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out

    
    @torch.inference_mode()
    def predict(self, dc: torch.Tensor, mask: Optional[np.ndarray], c_step: int, r_step: int, *args, **kwargs) -> torch.Tensor:
        raise RuntimeError("The datacube is currently not adapted to this model")
    
        prediction: torch.Tensor = torch.full((r_step * c_step,), fill_value=TDC.OUTPUT_NODATA, dtype=torch.long)
        if mask:
            merged_row: torch.Tensor = torch.full(c_step, fill_value=TDC.OUTPUT_NODATA, dtype=torch.long)
            for chunk_rows in range(0, r_step):
                merged_row.zero_()
                squeezed_row: torch.Tensor
                _, squeezed_row = torch.max(
                    self.forward(dc[chunk_rows, mask[chunk_rows]]).data,
                    dim=1
                )
                
                merged_row[mask[chunk_rows]] = squeezed_row
                prediction[chunk_rows, 0:c_step] = merged_row
        else:
            for chunk_rows in range(0, r_step):
                _, prediction[chunk_rows, 0:c_step] = torch.max(
                    self.forward(dc[chunk_rows]).data,
                    dim=1
                )
        
        return prediction


class LSTMCPU(nn.Module):
    def __init__(self, num_bands: int, input_size: int, hidden_size: int, num_layers: int, num_classes: int,
                 bidirectional: bool):
        super(LSTMCPU, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 1
        if bidirectional:
            self.D = 2
        self.embd = nn.Linear(num_bands, input_size)
        # definitions of forget gate, memory etc. are defined within nn.LSTM:
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(self.D * hidden_size, 256),  # hidden layer 1, amount of neurons in input layer get reduced to 256
            nn.ReLU(),  # activation function
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),  # dropping units at random to prevent overfitting
            nn.Linear(256, num_classes),  # hidden layer 2
            nn.Softmax(dim=1)  # final layer to calculate probablitiy for each class
        )

    def forward(self, x: Tensor):
        # x shape (batch, time_step, input_size)
        # out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = self.embd(x)
        h0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMRegression(nn.Module):
    def __init__(self, num_bands: int, input_size: int, hidden_size: int, num_layers: int, num_classes: int,
                 bidirectional: bool):
        super(LSTMRegression, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 1
        if bidirectional:
            self.D = 2
        self.embd = nn.Linear(num_bands, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(self.D * hidden_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, num_classes)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = self.embd(x)
        h0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out


class LSTMMultiLabel(nn.Module):
    def __init__(self, num_bands: int, input_size: int, hidden_size: int, num_layers: int, num_classes: int,
                 bidirectional: bool):
        super(LSTMMultiLabel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.D = 1
        if bidirectional:
            self.D = 2
        self.embd = nn.Linear(num_bands, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Sequential(
            nn.Linear(self.D * hidden_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # out shape (batch, num_classes)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        x = self.embd(x)
        h0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.D * self.num_layers, x.size(0), self.hidden_size).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out

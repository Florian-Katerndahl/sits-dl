"""
TRANSFORMER implementation adapted from https://github.com/JKfuberlin/SITST4TSC
"""
import torch
from torch import nn, Tensor
from torch.nn.modules.normalization import LayerNorm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import DataLoader
import math
from typing import Optional
import numpy as np
from sits_dl.tensordatacube import TensorDataCube as TDC

'''
This script defines an instance of the Transformer Object for Classification
It consists of two classes, one for Positional Encoding and another for the classifier itself
Both have a __init__ for initialization and a 'forward' method that is called during training and validation 
because they are both subclasses of nn.Module, and this is a required method in PyTorch's module system.

'''
dropout=0.1
max_len = 5000 # defining maximum sequence length the model will be able to process here
# model dimensions (hyperparameter), dropout is already included here to make the model less prone to overfitting due to a specific sequence, max_length is defined. DOY needs to be smaller than that
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# I don't know if this makes sense here, as device should be assigned in the main script where model is trained / inference happens

class PositionalEncoding(nn.Module):
    """Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        pos_encoder = PositionalEncoding(d_model)
    """
    def __init__(self, d_model:int, dropout=0.1, max_len=5000): # model dimensions (hyperparameter), dropout is already included here to make the model less prone to overfitting due to a specific sequence, max_length is defined. DOY needs to be smaller than that
        super(PositionalEncoding, self).__init__() # The super() builtin returns a proxy object (temporary object of the superclass) that allows us to access methods of the base class.
        # i do not understand what that means
        self.dropout = nn.Dropout(p=dropout) # WTF i do not understand, what this does
        pe = torch.zeros(max_len, d_model) # positional encoding object is initialized with zeros, according to max length and model dimension. 5000 because we need a position on the sin/cos line for every possible DOY
        positionPE = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) # used to create a 1-dimensional tensor representing the positions of tokens in a sequence.
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # a tensor representing the values used for scaling in the positional encoding calculation
        # Apply the sinusoidal encoding to even indices and cosine encoding to odd indices
        pe[:, 0::2] = torch.sin(positionPE * div_term)
        pe[:, 1::2] = torch.cos(positionPE * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1) # torch.Size([5000, 1, 204]) max_len, ?, d_model
        self.register_buffer('pe', pe) # WTF i don't know what and why

    def forward(self, doy):
        doy = doy.to(self.pe.device)
        return self.pe[doy, :]

class TransformerClassifier(nn.Module):
    def __init__(self, num_bands:int, num_classes:int, d_model:int, nhead:int, num_layers:int, dim_feedforward:int, sequence_length:int) -> None:
        super(TransformerClassifier, self).__init__()
        self.sequence_length = sequence_length
        self.d_model = d_model
        # encoder embedding, here a linear transformation is used to create the embedding, apparently it is an instance of nn.Linear that takes num_bands and d_model as args
        self.src_embd = nn.Linear(num_bands, d_model) # GPT: this linear transformation involves multiplying the input by a weight matrix and adding a bias vector.

        # transformer model
        encoder_layer = TransformerEncoderLayer(d_model*2, nhead, dim_feedforward, batch_first=True) # batch_first = True to avoid a warning concerning nested tensors and probably speeding up inference
        encoder_norm = LayerNorm(d_model*2)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        # classification
        self.fc = nn.Sequential( # does this run for each step of the sequence?
                    nn.Linear(d_model*2, 256), # condense the output of the pooling of the transformer encoder into a more dense representation
                    nn.ReLU(), # ReLu introduces non-linearity into the  network https://builtin.com/machine-learning/relu-activation-function
                    nn.BatchNorm1d(256), # speeds up and stabilizes by normalizing the previous activations
                    nn.Dropout(0.3), # randomly sets a fraction of the input to 0 to prevent overfitting during traninig, model.eval() should disable this
                    nn.Linear(256, num_classes), # condense to the number of classes
                    nn.Softmax(dim=1) # get activation percentage value for each class
                )

    def forward(self, input_sequence: Tensor) -> Tensor:
        """
        Forward pass of the TransformerClassifier.

        Parameters:
            input_sequence (torch.Tensor): Input sequence tensor of shape (seq_len, batch_size, num_bands).
            doy_sequence (torch.Tensor): Day-of-year sequence tensor of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """

        if len(input_sequence.shape) == 2:
            # Add a batch dimension if it's not present (assuming batch size of 1)
            input_sequence = input_sequence.unsqueeze(1)
        input_sequence_bands = input_sequence[:,:,0:10] # this is the input sequence without DOY
        obs_embed = self.src_embd(input_sequence_bands)  # [batch_size, seq_len, d_model] #
        self.PEinstance = PositionalEncoding(d_model=self.d_model, max_len=max_len)
        # this is where the input of form [batch_size, seq_len, n_bands] is passed through the linear transformation of the function src_embd()
        # to create the embeddings
        # Repeat obs_embed to match the shape [batch_size, seq_len, embedding_dim*2]
        x = obs_embed.repeat(1, 1, 2)
        # Add positional encoding based on day-of-year
        # X dimensions are [batch_size, seq_length, d_model*2], iterates over number of samples in each batch
        for i in range(input_sequence.size(0)):
            x[i, :, self.d_model:] = self.PEinstance(input_sequence[i, :, 10].long()).squeeze()
        # each batch's embedding is sliced and the second half replaced with a positional embedding of the DOY (11th column of the input_sequence) at the corresponding observation i

        output_encoder = self.transformer_encoder(x)
        # output: [batch_size, seq_len, d_model]
        # pool, _ = torch.max(output_encoder, dim=1, keepdim=False)
        meanpool = output_encoder.mean(dim=1) #this is global mean pooling   # [batch_size, seq_len, d_model]
        # maxpool = output_encoder.max(dim=1)
        # TODO: compare max and avg pooling
        output = self.fc(meanpool) # should be [batch_size, num_classes]
        # output2 = self.fc(maxpool)
        # final shape: [batch_size, num_classes]
        return output


    @torch.inference_mode()
    def predict(self, dc: torch.Tensor, mask: Optional[np.ndarray], c_step: int, r_step: int, batch_size: int, device: torch.device, *args, **kwargs) -> torch.Tensor:
        dl: DataLoader = TDC.to_dataloader(dc, batch_size)
        prediction: torch.Tensor = torch.full((r_step * c_step,), fill_value=TDC.OUTPUT_NODATA, dtype=torch.long)

        if mask is not None:
            mask_torch: torch.Tensor = torch.from_numpy(mask).bool()
            for batch_index, batch in enumerate(dl):
                for _, samples in enumerate(batch):
                    start: int = batch_index * batch_size
                    end: int = start + len(samples)
                    subset: torch.Tensor = mask_torch[start:end]
                    if not torch.any(subset):
                        next
                    input_tensor: torch.Tensor = samples[subset].to(device, non_blocking=True)  # ordering of subsetting and moving makes little to no difference time-wise but big difference memory-wise
                    _, data = torch.max(
                        self.forward(input_tensor),
                        dim=1
                    )
                    prediction[start:end][subset] = data.cpu()
        else:
            for batch_index, batch in enumerate(dl):
                for _, samples in enumerate(batch):
                    start: int = batch_index * batch_size
                    end: int = start + len(samples)
                    _, data = torch.max(
                        self.forward(samples.to(device, non_blocking=True)),
                        dim=1
                    )
                    prediction[start:end] = data.cpu()

        return torch.reshape(prediction, (r_step, c_step))

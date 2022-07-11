# Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding():
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_angle(self, position, i, d_model):
        angle = position / torch.pow(10000, (2 * (i//2) / d_model))
        return angle
    
    def positional_encoding(self, position, d_model):
        # Get the radians
        pos = torch.range(0, position - 1).to(device).view(-1, 1)
        i = torch.range(0, d_model - 1).to(device)
        rads = self.get_angle(pos, i, d_model)

        # Get the sinusoidal functions
        print(rads)
        pe_sin = torch.sin(rads[:, 0::2])
        pe_cos = torch.cos(rads[:, 1::2])
        
        # Get the positional encoding values
        pe = np.zeros(rads.shape)
        pe[:, 0::2] = pe_sin
        pe[:, 1::2] = pe_cos
        
        return pe
    
sample_pos_encoding = PositionalEncoding(50, 128)
plt.pcolormesh(sample_pos_encoding.pos_encoding, cmap='RdBu')
plt.xlabel('Depth')
plt.xlim((0, 128))
plt.ylabel('Position')
plt.colorbar()
plt.savefig("NLP_01_Transformer/check.png")
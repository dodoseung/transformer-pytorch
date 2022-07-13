# Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

# Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper parameter
d_model_value = 512
num_heads_value = 8

class PositionalEncoding():
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def positional_encoding(self, position, d_model):
        # Get the radians
        pos = torch.range(0, position - 1).to(device).view(-1, 1)
        i = torch.range(0, d_model - 1).to(device)
        rads = pos / torch.pow(10000, (2 * (i // 2) / d_model))

        # Get the positional encoding values from sinusoidal functions
        pe = np.zeros(rads.shape)
        pe[:, 0::2] = torch.sin(rads[:, 0::2])
        pe[:, 1::2] = torch.cos(rads[:, 1::2])
        
        return pe
    
# Query size: (batch, num_heads, query_sentence_len, d_model/num_heads)
# Key size: (batch, num_heads, key_sentence_len, d_model/num_heads)
# Value size: (batch, num_heads, value_sentence_len, d_model/num_heads)
# Mask size(optional): (batch, 1, 1, key_sentence_len)   
def scaled_dot_product_attention(queries, keys, values, mask):

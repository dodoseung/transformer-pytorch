# Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import matplotlib.pyplot as plt

import timeit

# Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Positional encoding
class PositionalEncoding():
    def __init__(self, position, d_model=512):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = self.positional_encoding(position, d_model)
    
    def positional_encoding(self, position, d_model):
        # Get the radians
        pos = torch.range(0, position - 1).to(device).view(-1, 1)
        i = torch.range(0, d_model - 1).to(device)
        rads = pos / torch.pow(10000, (2 * (i // 2) / d_model))

        # Get the positional encoding value from sinusoidal functions
        pe = np.zeros(rads.shape)
        pe[:, 0::2] = torch.sin(rads[:, 0::2])
        pe[:, 1::2] = torch.cos(rads[:, 1::2])
        
        return pe

# Encoder
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layer):
        super(Encoder, self).__init__()
        self.layers = [encoder_layer for _ in range(num_layer)]   
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, d_ff=2048):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model=d_model, dff=d_ff)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        out = self.multi_head_attention(x, x, x, mask)
        out = self.dropout(out)
        out = self.layer_norm(out)
        
        out = self.position_wise_feed_forward(x)
        out = self.dropout(out)
        out = self.layer_norm(out)
        
        return out

# Multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model / self.num_heads
        
        # Define w_q, w_k, w_v, w_o
        self.weight_q = nn.Linear(self.d_model, self.d_model)
        self.weight_k = nn.Linear(self.d_model, self.d_model)
        self.weight_v = nn.Linear(self.d_model, self.d_model)
        self.weight_o = nn.Linear(self.d_model, self.d_k)
    
    def forward(self, query, key, value, batch, mask=None):
        # (batch, seq_len, d_k) -> (batch, seq_len, d_model)
        query = self.weight_q(query)
        key = self.weight_k(key)
        value = self.weight_v(value)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        query = query.view(batch, -1, self.num_heads, self.d_k)
        key = key.view(batch, -1, self.num_heads, self.d_k)
        value = value.view(batch, -1, self.num_heads, self.d_k)
        
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = torch.transpose(query, 1, 2)
        key = torch.transpose(key, 1, 2)
        value = torch.transpose(value, 1, 2)
        
        # Unsqueeze the masks
        # (batch, seq_len, seq_len) -> (batch, 1, seq_len, seq_len)
        if mask is not None:
            mask = mask.unsqueeze(1)
        
        # Get the scaled attention
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k)
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = torch.transpose(scaled_attention, 1, 2)

        # Concat the splitted attentions
        # (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        concat_attention = torch.reshape(scaled_attention, (batch, -1, self.d_model))
        
        # Get the multi head attention
        # (batch, seq_len, d_model) -> (batch, seq_len, d_k)
        multihead_attention = self.weight_o(concat_attention)
        
        return multihead_attention
    
class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model=512, d_ff=2048):
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        
        return out
    
# Query, key, and value size: (batch, num_heads, seq_len, d_k)
# Mask size(optional): (batch, 1, seq_len, seq_len)   
def scaled_dot_product_attention(query, key, value, mask):
    # Get the q matmul k_t
    # (batch, h, seq_len, d_k) -> (batch, h, seq_len, seq_len)
    attention_score = torch.mm(query, torch.transpose(key, -2, -1))
    
    # Get the attention wights
    d_k = query.size(-1)
    attention_score = attention_score / math.sqrt(d_k)
    attention_score += (mask * -1e9) if mask is not None else 0
    attention_weights = F.softmax(attention_score, dim=-1, dtype=torch.float)
     
    # Get the attention value
    # (batch, h, seq_len, seq_len) -> (batch, h, seq_len, d_k)
    attention_value = torch.mm(attention_weights, value)
    
    return attention_value

def look_ahead_mask(size_pad):
    mask = torch.ones(size_pad, size_pad)
    mask = torch.triu(mask, diagonal=1)
    mask = mask.unsqueeze(0)
    print(mask.shape)
    return mask

def padding_mask():
    return 0

print(look_ahead_mask(5))

# t = timeit.Timer(lambda: scaled_dot_product_attention(temp_q, temp_k, temp_v, None)) 
# print (t.timeit(10000))
# print (t.timeit(10000))
# print (t.timeit(10000))



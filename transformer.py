# Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

import timeit

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

        # Get the positional encoding value from sinusoidal functions
        pe = np.zeros(rads.shape)
        pe[:, 0::2] = torch.sin(rads[:, 0::2])
        pe[:, 1::2] = torch.cos(rads[:, 1::2])
        
        return pe
    
# Query size: (batch, num_heads, query_sentence_len, d_model/num_heads)
# Key size: (batch, num_heads, key_sentence_len, d_model/num_heads)
# Value size: (batch, num_heads, value_sentence_len, d_model/num_heads)
# Mask size(optional): (batch, 1, 1, key_sentence_len)   
def scaled_dot_product_attention(query, key, value, mask):
    # Get the q matmul k_t
    q_dot_k = torch.mm(query, torch.t(key))
    
    # Get the attention wights
    # d_k = d_model / num_heads
    d_k = query.size(-1)
    logits = q_dot_k / d_k**(1/2)
    logits += (mask * -1e9) if mask is not None else 0
    attention_wights = F.softmax(logits, dim=-1, dtype=torch.float)
     
    # Get the attention value
    attention_value = torch.mm(attention_wights, value)
    
    return attention_wights, attention_value

class MultiHeadAttention():
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_v = self.d_model / self.num_heads
        
        # Define w_q, w_k, w_v, w_o
        self.weight_q = nn.Linear(self.d_model, self.d_model)
        self.weight_k = nn.Linear(self.d_model, self.d_model)
        self.weight_v = nn.Linear(self.d_model, self.d_model)
        self.weight_o = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, query, key, value, mask, batch_size):
        query = self.weight_q(query)
        key = self.weight_k(key)
        value = self.weight_v(value)
        
        # Split heads
        query = torch.reshape(query, (batch_size, self.num_heads, -1, self.d_v))
        key = torch.reshape(key, (batch_size, self.num_heads, -1, self.d_v))
        value = torch.reshape(value, (batch_size, self.num_heads, -1, self.d_v))
        
        # Get the scaled attention
        _, scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = scaled_attention.permute(0, 2, 1, 3)

        # Concat the splitted attentions
        concat_attention = torch.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        # Get the multi head attention
        multihead_attention = self.weight_o(concat_attention)
        
        return multihead_attention

def encoder(d_ff, d_model, num_heads, dropout):
    return 0

# t = timeit.Timer(lambda: scaled_dot_product_attention(temp_q, temp_k, temp_v, None)) 
# print (t.timeit(10000))
# print (t.timeit(10000))
# print (t.timeit(10000))



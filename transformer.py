# Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017).
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math

# Torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformer
class Transformer(nn.Module):
    def __init__(self, num_encoder_layer=6, num_decoder_layer=6, d_model=512, num_heads=8, d_ff=2048, vocab_size=10000):
        super(Transformer, self).__init__()
        # Layers
        encoder_layer = EncoderLayer(d_model, num_heads, d_ff)
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
        
        # Encoder and decoder
        self.encoder = Encoder(encoder_layer, num_encoder_layer)
        self.decoder = Decoder(decoder_layer, num_decoder_layer)
        
        # Output layer
        self.out_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, trg, src_mask, trg_mask):
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(trg, trg_mask, encoder_out, src_mask)
        out = self.out_layer(decoder_out)
        out = F.log_softmax(out, dim=-1)
        
        return out

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
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask):
        # Multi head attention
        out = self.multi_head_attention(x, x, x, mask)
        out = self.dropout(out)
        out = self.layer_norm(x + out)
        
        # Position wise feed foward
        out = self.position_wise_feed_forward(x)
        out = self.dropout(out)
        out = self.layer_norm(x + out)
        
        return out

# Decoder
class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layer):
        super(Decoder, self).__init__()
        self.layers = [decoder_layer for _ in range(num_layer)]   
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

# Decoder layer
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.masked_multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x, mask, encoder_out, encoder_mask):
        # Masked multi head attention
        out = self.masked_multi_head_attention(x, x, x, mask)
        out = self.dropout(out)
        out = self.layer_norm(x + out)
        
        # Multi head attention
        out = self.multi_head_attention(out, encoder_out, encoder_out, encoder_mask)
        out = self.dropout(out)
        out = self.layer_norm(x + out)
        
        # Position wise feed foward
        out = self.position_wise_feed_forward(x)
        out = self.dropout(out)
        out = self.layer_norm(x + out)
        
        return out

# Multi head attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = self.d_model // self.num_heads
        
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
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
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

# Set the look ahead mask
# (1, size_pad, size_pad)
def look_ahead_mask(seq):
    size_pad = seq.size(-1)
    mask = torch.ones(size_pad, size_pad)
    mask = torch.triu(mask, diagonal=1)
    
    # Set the padding mask
    mask[:, seq[0,:]==0] = 1
    mask = mask.unsqueeze(0)

    return mask

# Set the padding mask
# (1, size_pad, size_pad)
def padding_mask(seq):
    size_pad = seq.size(-1)
    mask = torch.zeros(size_pad, size_pad)
    mask[:, seq[0,:]==0] = 1
    mask = mask.unsqueeze(0)
    
    return mask

# Embedding
class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(TransformerEmbedding, self).__init__()
        embedding = Embedding(d_model, vocab)
        positional_encoding = PositionalEncoding(d_model, max_seq_len=5000)

        # Embedding
        self.embedding = nn.Sequential(embedding, positional_encoding)
    
    def forward(self, x):
        out = self.embedding(x)
        return out

# Vocab embedding
class Embedding(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(len(vocab), d_model)
        self.vocab = vocab
        self.d_model = d_model
    
    def forward(self, x):
        out = self.embedding(x) * math.sqrt(self.d_model)
        return out

# Positional encoding
class PositionalEncoding():
    def __init__(self, d_model=512, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()

        # Get the radians
        pos = torch.range(0, max_seq_len - 1).to(device).view(-1, 1)
        i = torch.range(0, d_model - 1).to(device)
        rads = pos / torch.pow(10000, (2 * (i // 2) / d_model))

        # Get the positional encoding value from sinusoidal functions
        pe = np.zeros(rads.shape)
        pe[:, 0::2] = torch.sin(rads[:, 0::2])
        pe[:, 1::2] = torch.cos(rads[:, 1::2])
        
        self.pe = pe.unsqueeze(0)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        out = Variable(self.pe[:, :x.size(1)], requires_grad=False).to(device)
        out = self.dropout(x + out)
        
        return out

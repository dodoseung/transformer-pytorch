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
    def __init__(self, num_encoder_layer=6, num_decoder_layer=6,
                 d_model=512, num_heads=8, d_ff=2048, dropout_rate=0.1,
                 vocab_size=10000):
        super(Transformer, self).__init__()
        # Embedding
        self.src_embedding = TransformerEmbedding(d_model, vocab_size)
        self.trg_embedding = TransformerEmbedding(d_model, vocab_size)
        
        # Layers
        decoder_layer = DecoderLayer(d_model, num_heads, d_ff)
        
        # Encoder and decoder
        self.encoder = Encoder(num_encoder_layer, d_model, num_heads, d_ff, dropout_rate)
        self.decoder = Decoder(decoder_layer, num_decoder_layer)
        
        # Output layer
        self.out_layer = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, trg):
        # Embedding
        src = self.src_embedding(src)
        trg = self.trg_embedding(trg)
        
        # Mask
        src_mask = padding_mask(src)
        trg_mask = look_ahead_mask(trg)
        
        # Encoder and decoder
        encoder_out = self.encoder(src, src_mask)
        decoder_out = self.decoder(trg, trg_mask, encoder_out, src_mask)
        
        # Transform to character
        out = self.out_layer(decoder_out)
        out = F.log_softmax(out, dim=-1)
        
        return out

# Encoder
class Encoder(nn.Module):
    def __init__(self, num_layer, d_model, num_heads, d_ff, dropout_rate, max_seq_len):
        super(Encoder, self).__init__()
        # Embedding
        self.src_embedding = TransformerEmbedding(d_model, max_seq_len)
        
        # Encoder layers
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout_rate) for _ in range(num_layer)])
        
    def forward(self, src, src_mask):
        # Embedding
        src = self.src_embedding(src)
        
        # Encoder layers
        for layer in self.layers:
            x = layer(src, src_mask)
            
        return x

# Encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout_ratio):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.position_wise_feed_forward = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff)
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, src, src_mask):
        # Multi head attention
        out1 = self.multi_head_attention(src, src, src, src_mask)
        out1 = self.dropout(out1)
        out1 = self.layer_norm(src + out1)
        
        # Position wise feed foward
        out2 = self.position_wise_feed_forward(out1)
        out2 = self.dropout(out2)
        out = self.layer_norm(out1 + out2)
        
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
        self.weight_o = nn.Linear(self.d_model, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        # Batch size
        batch_size = query.shape[0]
        
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        query = self.weight_q(query)
        key = self.weight_k(key)
        value = self.weight_v(value)
        
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k)
        query = query.view(batch_size, -1, self.num_heads, self.d_k)
        key = key.view(batch_size, -1, self.num_heads, self.d_k)
        value = value.view(batch_size, -1, self.num_heads, self.d_k)
        
        # (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = torch.transpose(query, 1, 2)
        key = torch.transpose(key, 1, 2)
        value = torch.transpose(value, 1, 2)
        
        # Get the scaled attention
        # (batch, h, query_len, d_k) -> (batch, query_len, h, d_k)
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = torch.transpose(scaled_attention, 1, 2)

        # Concat the splitted attentions
        # (batch, query_len, h, d_k) -> (batch, query_len, d_model)
        concat_attention = torch.reshape(scaled_attention, (batch_size, -1, self.d_model))
        
        # Get the multi head attention
        # (batch, query_len, d_model) -> (batch, query_len, d_model)
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
    # (batch, h, query_len, d_k) dot (batch, h, d_k, key_len)
    # -> (batch, h, query_len, key_len)
    attention_score = torch.mm(query, torch.transpose(key, -2, -1))
    
    # Get the attention score
    d_k = query.size(-1)
    attention_score = attention_score / math.sqrt(d_k)
    
    # Get the attention wights
    attention_score = attention_score.masked_fill(mask==0, -1e10) if mask is not None else attention_score
    attention_weights = F.softmax(attention_score, dim=-1, dtype=torch.float)
     
    # Get the attention value
    # (batch, h, query_len, key_len) -> (batch, h, query_len, d_k)
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
    print(seq[0,:]==0)
    mask[:, seq[0,:]==0] = 1
    mask = mask.unsqueeze(0)
    
    return mask

# Embedding
class TransformerEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size, max_seq_len=100):
        super(TransformerEmbedding, self).__init__()
        token_embedding = TokenEmbedding(d_model, vocab_size)
        position_embedding = PositionalEncoding(d_model, max_seq_len)

        # Embedding
        self.embedding = nn.Sequential(token_embedding, position_embedding)
    
    def forward(self, x):
        out = self.embedding(x)
        return out

# Token embedding
class TokenEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(TokenEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
    
    def forward(self, x):
        out = self.token_embedding(x) * math.sqrt(self.d_model)
        return out

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()

        # Get the radians
        pos = torch.range(0, max_seq_len - 1).to(device).view(-1, 1)
        i = torch.range(0, d_model - 1).to(device)
        rads = pos / torch.pow(10000, (2 * (i // 2) / d_model))

        # Get the positional encoding value from sinusoidal functions
        pe = torch.zeros(rads.shape)
        pe[:, 0::2] = torch.sin(rads[:, 0::2])
        pe[:, 1::2] = torch.cos(rads[:, 1::2])
        
        self.pe = pe.unsqueeze(0)
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x):
        seq_len = x.size(1)
        pos_enc = self.pe[:, :seq_len]
        
        out = x + Variable(pos_enc, requires_grad=False).to(device)
        out = self.dropout(out)
        
        return out

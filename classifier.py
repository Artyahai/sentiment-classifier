import torch
import torch.nn as nn 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int)->None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model, device=device)
    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int, dropout:float=0.1)->None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, h:int, dropout:float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model, device=device)
        self.w_k = nn.Linear(d_model, d_model, device=device)
        self.w_v = nn.Linear(d_model, d_model, device=device)
        self.w_o = nn.Linear(d_model, d_model, device=device)
        self.dropout = nn.Dropout(dropout)
    @staticmethod
    def attention(query, key, value, mask, dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, 1e-9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        return (attention_scores @ value), attention_scores
    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, d_model:int, eps:float=1e-6)->None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
    def forward(self, x:torch.Tensor):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias

class ResidualConnection(nn.Module):
    def __init__(self, dropout:float, d_model:int)->None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalization = LayerNormalization(d_model)
    def forward(self, x, sub_layer):
        return x + self.dropout(sub_layer(self.normalization(x)))


class FeedForwardBlock(nn.Module):
    def __init__(self, d_ff:int, d_model:int, dropout:float = 0.1)->None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff, device=device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model, device=device)
    def forward(self, x):
        return self.linear_2(self.dropout(self.relu(self.linear_1(x)))) 


class EncoderBlock(nn.Module):
    def __init__(self, d_model:int,h:int, d_ff:int, dropout:float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = FeedForwardBlock(d_ff, d_model, dropout)
        self.residual_1 = ResidualConnection(dropout, d_model)
        self.residual_2 = ResidualConnection(dropout, d_model)
    def forward(self, x, mask):
        x = self.residual_1(x,lambda x:self.self_attention(x,x,x,mask))
        x = self.residual_2(x,self.feed_forward)
        return x 


class Classifier(nn.Module):
    def __init__(self, d_model:int, vocab_size:int, seq_len:int, h:int, d_ff:int, num_classes:int, num_layers:int = 6, dropout:float = 0.1) -> None:
        super().__init__()
        self.embeddings = InputEmbeddings(d_model, vocab_size)
        self.pos_encoding = PositionalEncoding(d_model, seq_len, dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, h, d_ff, dropout) for _ in range(num_layers)])
        self.f_c = nn.Linear(d_model, num_classes)
    def forward(self, x, mask=None):
        x = self.embeddings(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = x.mean(dim=1)
        x = self.f_c(x)
        return x
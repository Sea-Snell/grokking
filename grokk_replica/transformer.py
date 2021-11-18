import torch.nn as nn
import torch.nn.functional as F
import torch
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_dim, attn_dim, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.attn_dim = attn_dim
        self.dropout = nn.Dropout(p=dropout)
        self.key_proj = nn.Linear(hidden_dim, self.attn_dim*self.heads)
        self.val_proj = nn.Linear(hidden_dim, self.attn_dim*self.heads)
        self.query_proj = nn.Linear(hidden_dim, self.attn_dim*self.heads)
        self.output_proj = nn.Linear(self.attn_dim*self.heads, hidden_dim)

    def forward(self, queries, keys, values, mask, past_kv=None):
        assert keys.shape[1] == values.shape[1], 'keys and values time dimension must match'
        assert past_kv is None or past_kv[0].shape[1] == past_kv[1].shape[1], 'cached keys and values time dimension must match'
        # queries/keys/values = (batch, time, hidden_dim)
        # mask = (batch, query_time, key_time) - bool tensor, True if should mask
        # past_kv = tuple of (past_k=(batch, time, head, hidden_dim), past_v=(batch, time, head, hidden_dim)) or None
        # returns:
        # attn_matrix = (batch, head, query_time, key_time)
        # attn_output = (batch, query_time, hidden_dim)
        # tuple of updated past_kv

        batch, time, _ = queries.shape
        key_heads = self.key_proj(keys).reshape(batch, time, self.heads, self.attn_dim)
        val_heads = self.val_proj(values).reshape(batch, time, self.heads, self.attn_dim)
        query_heads = self.query_proj(values).reshape(batch, time, self.heads, self.attn_dim)
        if past_kv is not None:
            past_k, past_v = past_kv
            key_heads = torch.cat([past_k, key_heads], dim=1)
            val_heads = torch.cat([past_v, val_heads], dim=1)
        attn_matrix = F.softmax((torch.einsum('bqhd,bkhd->hbqk', query_heads, key_heads)
                                 / math.sqrt(self.attn_dim)).masked_fill(mask, float('-inf')), dim=-1)
        attn_matrix = self.dropout(attn_matrix.transpose(0, 1).contiguous())
        combined_vals = torch.einsum('bkhd,bhqk->bqhd', val_heads, attn_matrix).reshape(batch, time, self.attn_dim*self.heads)
        attn_output = self.output_proj(combined_vals)
        return attn_output, attn_matrix, (key_heads, val_heads)

class TransformerBlock(nn.Module):
    def __init__(self, heads, hidden_dim, attn_dim, intermediate_dim, dropout=0.1, pre_norm=True):
        super(TransformerBlock, self).__init__()
        self.pre_norm = pre_norm
        self.attn = MultiHeadAttention(heads, hidden_dim, attn_dim, dropout=dropout)
        self.ff1 = nn.Linear(hidden_dim, intermediate_dim)
        self.ff2 = nn.Linear(intermediate_dim, hidden_dim)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, x, attn_mask, past_kv=None):
        if not self.pre_norm:
            attn_output, attn_matrix, past_kv = self.attn(x, x, x, attn_mask, past_kv=past_kv)
            x = self.layer_norm1(self.dropout1(attn_output) + x)
            mlp_out = self.ff2(self.dropout2(F.gelu(self.ff1(x))))
            x = self.layer_norm2(self.dropout3(mlp_out) + x)
        else:
            x_norm1 = self.layer_norm1(x)
            attn_output, attn_matrix, past_kv = self.attn(x_norm1, x_norm1, x_norm1, attn_mask, past_kv=past_kv)
            x = self.dropout1(attn_output) + x
            x_norm2 = self.layer_norm2(x)
            mlp_out = self.ff2(self.dropout2(F.gelu(self.ff1(x_norm2))))
            x = self.dropout3(mlp_out) + x
        return x, attn_matrix, past_kv

class Transformer(nn.Module):
    def __init__(self, vocab_size, max_length, heads, hidden_dim, attn_dim, intermediate_dim, num_blocks, block_repeats, output_size, dropout=0.1, pre_norm=True):
        super(Transformer, self).__init__()
        self.pre_norm = pre_norm
        self.hidden_dim = hidden_dim
        self.block_repeats = block_repeats
        self.max_length = max_length
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(heads, hidden_dim, attn_dim, intermediate_dim, dropout=dropout, pre_norm=pre_norm) for _ in range(num_blocks)
        ])
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.positions = nn.Embedding(max_length, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_size)
        self.dropout = nn.Dropout(p=dropout)
        if self.pre_norm:
            self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, attn_mask, past_kvs=None):
        # x = (batch, time)
        # attn_mask = (batch, query_time, key_time)
        # past_kvs = list of past_kvs for each layer

        attns = []
        new_past_kvs = []
        initial_pos = 0
        if past_kvs is not None:
            initial_pos = past_kvs[0][0].shape[1]
        assert initial_pos+x.shape[1] <= self.max_length, 'sequence too long'
        x = self.dropout(self.embeddings(x) * math.sqrt(self.hidden_dim) + self.positions.weight[initial_pos:initial_pos+x.shape[1], :])
        step = 0
        for _ in range(self.block_repeats):
            for i in range(len(self.transformer_blocks)):
                x, attn, past_kv = self.transformer_blocks[i](x, attn_mask, past_kv=past_kvs[step] if past_kvs is not None else None)
                attns.append(attn)
                new_past_kvs.append(past_kv)
                step += 1
        if self.pre_norm:
            x = self.norm(x)
        return self.output(x), attns, new_past_kvs

def xavier_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

def kaiming_init(model):
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.kaiming_uniform_(p, a=math.sqrt(5))

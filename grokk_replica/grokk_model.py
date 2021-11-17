import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import causal_attn_mask

class GrokkModel(nn.Module):
    def __init__(self, transformer):
        super(GrokkModel, self).__init__()
        self.transformer = transformer
    
    def forward(self, x):
        attn_mask = causal_attn_mask(x.shape[1]).unsqueeze(0).repeat(x.shape[0], 1, 1)
        predictions, _, _ = self.transformer(x, attn_mask)
        return predictions
    
    def get_loss(self, x):
        predictions = self(x[:, :-1])
        loss = F.cross_entropy(predictions.reshape(-1, predictions.shape[-1]), x[:, 1:].reshape(-1))
        accuracy = (torch.argmax(predictions[:, -1, :], dim=-1) == x[:, -1]).float().mean()
        return loss, {'loss': (loss.item(), x.shape[0]*(x.shape[1]-1)), 'accuracy': (accuracy.item(), x.shape[0])}
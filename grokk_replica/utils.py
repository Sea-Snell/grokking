import os
import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict
import math

def causal_attn_mask(seq_len, device=torch.device('cpu')):
    # seq_len = length of sequence
    # returns: (seq_len, seq_len)
    return torch.tensor(np.triu(np.ones((seq_len, seq_len)), k=1) == 1).to(device)

def combine_logs(logs):
    combined_logs = defaultdict(float)
    count_logs = defaultdict(float)
    for log in logs:
        for k, (v, c) in log.items():
            combined_logs[k] += v * c
            count_logs[k] += c
    return {k: combined_logs[k] / count_logs[k] for k in combined_logs.keys()}

def parameter_norm(model: nn.Module):
    norm = 0.0
    for param in model.parameters():
        norm += (param.norm() ** 2).item()
    return math.sqrt(norm)

def convert_path(path):
    if path is None:
        return None
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), '../', path)
import torch
import numpy as np
from collections import defaultdict

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


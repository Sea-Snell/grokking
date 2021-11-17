import torch
from torch.utils import data
from torch.utils.data import IterableDataset, Dataset
from datasets import AbstractDataset, ModSumDataset, ModSubtractDataset
from utils import combine_logs
from transformer import Transformer, xavier_init
from torch.utils.data import DataLoader
import torch.nn as nn
from grokk_model import GrokkModel
from tqdm.auto import tqdm

class GroupDataset(IterableDataset):
    def __init__(self, dataset: AbstractDataset, split: str):
        super(GroupDataset, self).__init__()
        assert split in {'train', 'val'}
        self.dataset = dataset
        self.split = split
        self.fetch_f = None
        if self.split == 'train':
            self.fetch_f = self.dataset.fetch_train_example
        elif self.split == 'val':
            self.fetch_f = self.dataset.fetch_val_example
        else:
            raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        example, _ = self.fetch_f()
        return torch.tensor(example)

if __name__ == "__main__":
    dataset = ModSumDataset(97, 0.1)
    train_data = GroupDataset(dataset, 'train')
    val_data = GroupDataset(dataset, 'val')
    transformer = Transformer(vocab_size=dataset.n_vocab, max_length=5, 
                              heads=4, hidden_dim=128, attn_dim=32, 
                              intermediate_dim=512, num_blocks=2, 
                              block_repeats=1, output_size=dataset.n_vocab, 
                              dropout=0.1, pre_norm=True)
    # xavier_init(transformer)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    grokk_model = GrokkModel(transformer).to(device)
    train_dataloader = DataLoader(train_data, num_workers=0, batch_size=512)
    val_dataloader = DataLoader(val_data, num_workers=0, batch_size=512)
    optim = torch.optim.AdamW(transformer.parameters(), lr=1e-3, weight_decay=1.0, 
                              betas=(0.9, 0.98))
    val_batches = 8
    step = 0
    for item in tqdm(train_dataloader):
        loss, logs = grokk_model.get_loss(item.to(device))
        optim.zero_grad()
        loss.backward()
        optim.step()
        if (step+1) % 100 == 0:
            all_val_logs = []
            for i, val_item in tqdm(enumerate(val_dataloader)):
                if i >= val_batches:
                    break
                _, val_logs = grokk_model.get_loss(val_item.to(device))
                all_val_logs.append(val_logs)
            print(step+1, {'val': combine_logs(all_val_logs), 'train': combine_logs([logs])})
        step += 1
        
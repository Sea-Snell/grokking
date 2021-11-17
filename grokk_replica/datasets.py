import abc
import random

class AbstractDataset(abc.ABC):
    def __init__(self, group_elements, frac_train):
        self.frac_train = frac_train
        self.group_elements = group_elements
        self.ordered_group_elements = list(self.group_elements)
        self.idx2vocab = ['o', '='] + self.ordered_group_elements
        self.vocab2idx = {vocab: idx for idx, vocab in enumerate(self.idx2vocab)}
        self.n_vocab = len(self.idx2vocab)
        idxs = list(range(len(self.group_elements)**2))
        random.shuffle(idxs)
        self.train_pairs, self.val_pairs = idxs[:int(len(idxs)*frac_train)], idxs[int(len(idxs)*frac_train):]
    
    @abc.abstractmethod
    def fetch_output(self, a, b):
        pass

    def encode(self, sequence):
        return [self.vocab2idx[item] for item in sequence]
    
    def decode(self, sequence):
        return [self.idx2vocab[item] for item in sequence]
    
    def form_equation(self, a, b, c):
        return [a, 'o', b, '=', c]
    
    def fetch_example(self, idx):
        a = self.ordered_group_elements[idx // len(self.group_elements)]
        b = self.ordered_group_elements[idx % len(self.group_elements)]
        c = self.fetch_output(a, b)
        equation = self.form_equation(a, b, c)
        return self.encode(equation), equation
    
    def fetch_train_example(self):
        idx = random.choice(self.train_pairs)
        return self.fetch_example(idx)

    def fetch_val_example(self):
        idx = random.choice(self.val_pairs)
        return self.fetch_example(idx)

class ModSumDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModSumDataset, self).__init__(list(range(p)), frac_train)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a + b) % self.p

class ModSubtractDataset(AbstractDataset):
    def __init__(self, p, frac_train):
        super(ModSumDataset, self).__init__(list(range(p)), frac_train)
        self.p = p
    
    def fetch_output(self, a, b):
        return (a - b) % self.p


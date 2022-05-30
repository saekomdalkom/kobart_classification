# extends Lightning Data Module
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


class MyDataModule(pl.LightningDataModule):
    def __init__(self, 
                 train_file,
                 test_file, 
                 tokenizer,
                 max_len=512,
                 batch_size=10,
                 num_workers=5):
        super().__init__()
        self.batch_size = batch_size
        self.max_len = max_len
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tokenizer = tokenizer
        self.num_workers = num_workers

    # OPTIONAL, called for every GPU/machine (assigning state is OK)
    def setup(self, stage):
        # split dataset
        self.trainDataset = MyDataset(self.train_file_path,
                                 self.tokenizer,
                                 self.max_len)
        self.testDataset = MyDataset(self.test_file_path,
                                self.tokenizer,
                                self.max_len)

    def train_dataloader(self):
        train = DataLoader(self.trainDataset,
                           batch_size=self.batch_size,
                           num_workers=self.num_workers, 
                           shuffle=True)
        return train

    def val_dataloader(self):
        val = DataLoader(self.testDataset,
                         batch_size=self.batch_size,
                         num_workers=self.num_workers, 
                         shuffle=False)
        return val

    def test_dataloader(self):
        test = DataLoader(self.testDataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers, 
                          shuffle=False)
        return test



class MyDataset(Dataset):
    def __init__(self, file, tokenizer, max_len, pad_index = 0, ignore_index=-100):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep='\t')
        self.len = self.docs.shape[0]
        self.pad_index = pad_index
        self.ignore_index = ignore_index
    
    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]
        decision, label = str(instance['decision']), int(instance['label'])

        # print("asd", decision, label)
        # input_id = self.tokenizer.encode(decision)
        tokens = [self.tokenizer.bos_token] + \
            self.tokenizer.tokenize(decision) + [self.tokenizer.eos_token]
        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_id)
        if len(input_id) < self.max_len:
            while len(input_id) < self.max_len:
                input_id += [self.tokenizer.pad_token_id]
                attention_mask += [0]
        else:
            input_id = input_id[:self.max_len - 1] + [self.tokenizer.eos_token_id]
            attention_mask = attention_mask[:self.max_len]
        return {'input_ids': np.array(input_id, dtype=np.int_),
                'attention_mask': np.array(attention_mask, dtype=np.float),
                'labels': np.array(label, dtype=np.int_)}
    
    def __len__(self):
        return self.len

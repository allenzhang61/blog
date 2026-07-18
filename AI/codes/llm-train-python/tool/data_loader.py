import tiktoken
from torch.utils.data import Dataset, DataLoader
import torch


def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True,
                         drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=0)
    return dataloader


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)  # A

        for i in range(0, len(token_ids) - max_length, stride):  # B
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __len__(self):  # C
        return len(self.input_ids)


    def __getitem__(self, idx):  # D
        return self.input_ids[idx], self.target_ids[idx]

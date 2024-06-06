import os
import json
import random
from torch.utils.data import Dataset

class GADataset(Dataset):
    def __init__(self, data, tokenizer, min_len=50, max_len=2000):
        self.data = data
        self.tokenizer = tokenizer
        self.min_len = min_len
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # tokenize the text. The objective is next tokem prediction
        return self.data[idx]

    def collate_fn(self, batch):
        inputs = self.tokenizer(batch, padding="max_length", truncation=True, max_length=self.max_len, return_tensors="pt")
        return inputs


def get_ga_data(forget_corpora, retain_corpora, tokenizer, min_len=50, max_len=2000, batch_size=1):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            from datasets import load_dataset
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        else:
            current_dir_path = os.path.dirname(os.path.realpath(__file__))
            for line in open(f"{current_dir_path}/wmdp/data/{name}.jsonl", "r"):
                if "bio-forget-corpus" in name:
                    raw_text = json.loads(line)['text']
                else:
                    raw_text = line
                if len(raw_text) > min_len:
                    data.append(str(raw_text))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    # combine lists of datasets
    raw_forget_set = [get_dataset(c) for c in forget_corpora]
    raw_forget_set = [item[0] for sublist in raw_forget_set for item in sublist]
    random.shuffle(raw_forget_set)

    # split raw_forget_set intro train and test
    train_size = int(0.8 * len(raw_forget_set))
    train_set = raw_forget_set[:train_size]
    test_set = raw_forget_set[train_size:]
    return GADataset(train_set, tokenizer, min_len=min_len, max_len=max_len), GADataset(test_set, tokenizer, min_len=min_len, max_len=max_len)
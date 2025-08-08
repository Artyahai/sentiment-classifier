from transformers import AutoTokenizer
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
tokenizer  = AutoTokenizer.from_pretrained('bert-base-uncased')
dataset = load_dataset('sentiment140', split='train')
small_dataset = dataset.shuffle(seed=42).select(range(10000))


def transform_labels(example):
    example['sentiment'] = 1 if example['sentiment'] == 4 else 0
    return example

small_dataset = small_dataset.map(transform_labels)

def tokenize_func(examples):
    return tokenizer(examples['text'], padding='max_length', truncation = True, max_length = 100)

small_dataset = small_dataset.map(tokenize_func)

input_ids = torch.tensor(small_dataset['input_ids'])
attention_masks = torch.tensor(small_dataset['attention_mask'])
labels = torch.tensor(small_dataset['sentiment'])

dataset = TensorDataset(input_ids, attention_masks, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
batch_size = 32
train_dataset, test_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size = batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

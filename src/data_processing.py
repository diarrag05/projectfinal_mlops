from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def preprocess_data(data):
    data['text'] = data['text'].str.lower().str.replace(r'[^a-z0-9 ]', '', regex=True)
    train_df, val_df = train_test_split(data, test_size=0.2)
    train_enc = tokenizer(train_df['text'].tolist(), padding=True, truncation=True, max_length=512)
    val_enc = tokenizer(val_df['text'].tolist(), padding=True, truncation=True, max_length=512)
    return SentimentDataset(train_enc, train_df['label'].tolist()), SentimentDataset(val_enc, val_df['label'].tolist())

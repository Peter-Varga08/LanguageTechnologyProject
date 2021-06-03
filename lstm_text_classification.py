# library imports
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
import torch.nn.functional as F
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from dataloader_movies import movies_df, label_encoder, shortlisted_genres


class TextClassifierLSTM(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.2, num_layers=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # self.dropout = nn.Dropout(0.3)
        self.embeddings = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=self.num_layers)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, s):
        # Hidden and cell state definion
        h = torch.zeros((self.num_layers, x.size(0), self.hidden_dim))
        c = torch.zeros((self.num_layers, x.size(0), self.hidden_dim))

        # Initialization fo hidden and cell states
        torch.nn.init.xavier_normal_(h)
        torch.nn.init.xavier_normal_(c)

        x = self.embeddings(x)
        # x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack, (h, c))
        output, _ = pad_packed_sequence(out_pack, batch_first=True)
        # out = self.linear(ht[-1])
        out = self.fc(output)
        out = F.log_softmax(output, dim=1)
        return out


class PlotsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]


# |--------------------------------------|
# |          Pre-processing              |
# |--------------------------------------|
def tokenize(text, tokenizer):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    regex = re.compile('[' + re.escape(string.punctuation) + '0-9\\r\\t\\n]')  # remove punctuation and numbers
    nopunct = regex.sub(" ", text.lower())
    return [token.text for token in tokenizer.tokenizer(nopunct)]


def trim_rare_words(plots, tokenizer):
    # count number of occurences of each word
    counts = Counter()
    for index, row in plots.iterrows():
        counts.update(tokenize(row['Plot'], tokenizer))
    # deleting infrequent words
    print("num_words before:", len(counts.keys()))
    for word in list(counts):
        if counts[word] < MIN_COUNT:
            del counts[word]
    print("num_words after:", len(counts.keys()))
    return counts


def create_vocabulary(counts):
    # creating vocabulary
    vocab2index = {"": 0, "UNK": 1}
    words = ["", "UNK"]
    for word in counts:
        vocab2index[word] = len(words)
        words.append(word)
    return vocab2index


# TODO: Adjust sentence length to max_sentence_length instead of N=70
def encode_sentence(text, tokenizer, vocab2index, N=70):
    tokenized = tokenize(text, tokenizer)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


# |--------------------------------------|
# |       Training & Evaluation          |
# |--------------------------------------|
# TODO: Make function to utilize GPU instead of CPU is GPU exists
def train_model(model, epochs=10):
    model.train()
    print("Training classifier...")
    for i in range(epochs):
        sum_loss = 0.0
        total = 0
        correct = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            # TODO: fix target size bug
            loss = F.cross_entropy(y_pred, y)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
            correct += (y_pred == y).float().sum()
            print(f"Epoch [{i}]: train loss {(sum_loss/total)}, train_acc: {correct/total}")
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)
        if i % 5 == 0:
            print("train loss %.3f, val loss %.3f, val accuracy %.3f, and val rmse %.3f" % (sum_loss/total, val_loss, val_acc, val_rmse))


def validation_metrics (model, valid_dl):
    model.eval()
    correct = 0
    total = 0
    sum_loss = 0.0
    sum_rmse = 0.0
    for x, y, l in valid_dl:
        x = x.long()
        y = y.long()
        y_hat = model(x, l)
        loss = F.cross_entropy(y_hat, y)
        pred = torch.max(y_hat, 1)[1]
        correct += (pred == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        sum_rmse += np.sqrt(mean_squared_error(pred, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total


MIN_COUNT = 3
# ----------- Model variables -----------
BATCH_SIZE = 16
HIDDEN_DIM = 50
EMBEDDING_DIM = 50

tok = spacy.load('en_core_web_sm')
counts = trim_rare_words(movies_df, tok)
word2idx = create_vocabulary(counts)
movies_df['Plot_encoded'] = movies_df['Plot'].apply(lambda x: np.array(encode_sentence(x, tok, word2idx)))
num_classes = len(shortlisted_genres)

X = list(movies_df['Plot_encoded'])
y = list(movies_df['Genre_encoded'])
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)
train_ds = PlotsDataset(X_train, y_train)
valid_ds = PlotsDataset(X_valid, y_valid)
train_dl = DataLoader(train_ds)
val_dl = DataLoader(valid_ds)

model = TextClassifierLSTM(len(word2idx), EMBEDDING_DIM, HIDDEN_DIM, num_classes)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=0.05)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(model, epochs=10)

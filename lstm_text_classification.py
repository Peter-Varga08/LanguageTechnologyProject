import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
import torch.nn.functional as F
import string
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from dataloader_movies import movies_df, label_encoder, shortlisted_genres
from annoy import AnnoyIndex
from spacy.lang.en.examples import sentences


class PreTrainedEmbeddings(object):
    def __init__(self, word_to_index, word_vectors):
        """
        Args:
        word_to_index (dict): mapping from word to integers
        word_vectors (list of numpy arrays)
        """
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors
        self.index_to_word = {v: k for k, v in self.word_to_index.items()}
        self.index = AnnoyIndex(len(word_vectors[0]), metric='euclidean')
        for _, i in self.word_to_index.items():
            self.index.add_item(i, self.word_vectors[i])
        self.index.build(50)

    @classmethod
    def from_embeddings_file(cls, embedding_file):
        """Instantiate from pretrained vector file.
        Vector file should be of the format:
        word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
        word1 x1_0 x1_1 x1_2 x1_3 ... x1_N
        Args:
        embedding_file (str): location of the file
        Returns:
        instance of PretrainedEmbeddings
        """

        word_to_index = {}
        word_vectors = []
        with open(embedding_file) as fp:
            for line in fp.readlines():
                line = line.split(" ")
                word = line[0]
                vec = np.array([float(x) for x in line[1:]])
                word_to_index[word] = len(word_to_index)
                word_vectors.append(vec)
        return cls(word_to_index, word_vectors)

    def get_embedding(self, word):
        """
        Args:
        word (str)
        Returns
        an embedding (numpy.ndarray)
        """

        return self.word_vectors[self.word_to_index[word]]

    def get_closest_to_vector(self, vector, n=1):
        """Given a vector, return its n nearest neighbors
        Args:
        vector (np.ndarray): should match the size of the vectors
        in the Annoy index
        n (int): the number of neighbors to return
        Returns:
        [str, str, ...]: words nearest to the given vector
        The words are not ordered by distance
        """
        nn_indices = self.index.get_nns_by_vector(vector, n)
        return [self.index_to_word[neighbor]
                for neighbor in nn_indices]


class TextClassifierLSTM(torch.nn.Module):
    def __init__(self, vocab, embedding_dim, hidden_dim, num_classes, dropout=0.2, num_layers=2):
        super().__init__()
        # --------------------- Adding a Pretrained embedding layer to the network ---------------------
        assert embedding_dim in [50, 100, 200, 300]  # GloVe pre-trained embeddings only contain these dimensions
        self.embedding_dim = embedding_dim
        self.glove_vocab = PreTrainedEmbeddings.from_embeddings_file(f'./data/glove.6B.{self.embedding_dim}d.txt')
        self.vocab = vocab
        self.embedding_weights = np.zeros((len(self.vocab), self.embedding_dim))
        for i, word in enumerate(self.vocab):
            try:
                self.embedding_weights[i] = self.glove_vocab.get_embedding(word)
            except KeyError:
                self.embedding_weights[i] = np.random.normal(scale=0.6, size=(self.embedding_dim,))
        self.embedding_weights = torch.FloatTensor(self.embedding_weights)
        self.embeddings = nn.Embedding.from_pretrained(self.embedding_weights, freeze=True, padding_idx=0)
        # ------------------------------------------------------------------------------------------------------------
        self.vocab_size = len(vocab)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        # self.embeddings = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True,
                            dropout=dropout, num_layers=self.num_layers, bidirectional=True)
        # self.fc = nn.Linear(2*hidden_dim, num_classes)
        self.fc = nn.Linear(hidden_dim, num_classes)


    def forward(self, x, s):
        x = self.embeddings(x)
        x = self.dropout(x)
        x_pack = pack_padded_sequence(x, s, batch_first=True, enforce_sorted=False)
        out_pack, (ht, ct) = self.lstm(x_pack)
        out = self.fc(ht[-1])

        return out


class PlotsDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.y = Y
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx][0].astype(np.int32)), self.y[idx], self.X[idx][1]
        # return torch.Tensor(self.data[idx]).long(), torch.Tensor(self.labels[idx])


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
def encode_sentence(text, tokenizer, vocab2index, N=500):
    tokenized = tokenize(text, tokenizer)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


# |--------------------------------------|
# |       Training & Evaluation          |
# |--------------------------------------|
# TODO: Make function to utilize GPU instead of CPU if GPU exists
def train_model(model,train_dl, epochs=10):
    model.train()
    print("Training classifier...")
    for epoch in range(epochs):
        sum_loss = 0.0
        total = 0
        correct = 0
        for x,y,l in train_dl:
            x = x.long()
            y = y.long()
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            preds = y_pred.argmax(dim=-1)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item()*y.shape[0]
            total += y.shape[0]
            
            correct += (preds == y).float().sum()
            # print(f"Epoch [{epoch}]: train loss {(sum_loss/total)}, train_acc: {correct/total}")
        val_loss, val_acc, val_rmse = validation_metrics(model, val_dl)

        # if epoch % 5 == 0:
        print(f"Epoch [{epoch}]:"
              f" train loss {round(float(sum_loss/total), 4)},"
              f" train_acc: {round(float(correct/total), 4)}",
              f" val loss {round(float(val_loss), 4)},"
              f" val accuracy {round(float(val_acc), 4)}")
        # print("Epoch [%d], train loss %.3f, , and val rmse %.3f" % (epoch, sum_loss/total, val_loss, val_acc, val_rmse))


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
        preds = y_hat.argmax(dim=-1)
        correct += (preds == y).float().sum()
        total += y.shape[0]
        sum_loss += loss.item()*y.shape[0]
        # sum_rmse += np.sqrt(mean_squared_error(preds, y.unsqueeze(-1)))*y.shape[0]
    return sum_loss/total, correct/total, sum_rmse/total


MIN_COUNT = 3
# ----------- Model variables -----------
BATCH_SIZE = 16
HIDDEN_DIM = 100
EMBEDDING_DIM = 100

tok = spacy.load('en_core_web_sm')
counts = trim_rare_words(movies_df, tok)
word2idx = create_vocabulary(counts)
movies_df['Plot_encoded'] = movies_df['Plot'].apply(lambda x: np.array(encode_sentence(x, tok, word2idx)))
num_classes = len(shortlisted_genres)
print("number of classes %d" % num_classes)
X = list(movies_df['Plot_encoded'])
y = list(movies_df['genre_encoded'])

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2)

train_ds = PlotsDataset(X_train, y_train)
valid_ds = PlotsDataset(X_valid, y_valid)

train_dl= DataLoader(train_ds,
    shuffle=True,
    batch_size=BATCH_SIZE)

# train_dl = DataLoader()
val_dl = DataLoader(valid_ds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = TextClassifierLSTM(word2idx, EMBEDDING_DIM, HIDDEN_DIM, num_classes)
parameters = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(parameters, lr=5e-3, weight_decay=5e-4)
# optimizer = torch.optim.RMSprop(parameters, lr=5e-3, weight_decay=1e-3)
train_model(model, train_dl, epochs=15)

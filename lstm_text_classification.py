import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import re
import spacy
import torch.nn.functional as F
import string
import csv
import os
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import mean_squared_error
from collections import Counter
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

from dataloader_movies import movies_df, shortlisted_genres

# ----------- Pre-processing constants -----------
MIN_COUNT = 3
PLOT_LENGTH = 200

class PreTrainedEmbeddings(object):
    def __init__(self, word_to_index, word_vectors):
        """
        Args:
        word_to_index (dict): mapping from word to integers
        word_vectors (list of numpy arrays)
        """
        self.word_to_index = word_to_index
        self.word_vectors = word_vectors

    @classmethod
    def from_embeddings_file(cls, embedding_file):
        """Instantiate from pretrained vector file.
        Vector file should be of the format:
        word0 x0_0 x0_1 x0_2 x0_3 ... x0_N
        word1 x1_0 x1_1 x1_2 x1_3 ... x1_N
        Args:
        embedding_file (str): location of the file
        Returns: instance of PretrainedEmbeddings
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
        Returns an embedding (numpy.ndarray)
        """
        return self.word_vectors[self.word_to_index[word]]


class TextClassifierLSTM(torch.nn.Module):
    def __init__(self, vocab, embedding_dim, hidden_dim, num_classes, dropout=0.2, num_layers=2):
        super().__init__()
        # --------------------- Adding a Pretrained embedding layer to the network ---------------------
        assert embedding_dim in [50, 100, 200, 300]  # GloVe pre-trained embeddings only contain these dimensions
        self.embedding_dim = embedding_dim
        self.glove_vocab = PreTrainedEmbeddings.from_embeddings_file(
            f'./data/glove.6B.{self.embedding_dim}d.txt')
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
        # self.embeddings = nn.Embedding(self.vocab_size, embedding_dim, padding_idx=0)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
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
    if type(plots) == pd.core.frame.DataFrame:
        for index, row in plots.iterrows():
            counts.update(tokenize(row['Plot'], tokenizer))
    elif type(plots) == list:
        for row in plots:
            counts.update(tokenize(row, tokenizer))
    else:
        raise TypeError("Parameter 'plots' has to be either a non-nested list or a pandas Dataframe.")
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


def encode_sentence(text, tokenizer, vocab2index, N):
    tokenized = tokenize(text, tokenizer)
    encoded = np.zeros(N, dtype=int)
    enc1 = np.array([vocab2index.get(word, vocab2index["UNK"]) for word in tokenized])
    length = min(N, len(enc1))
    encoded[:length] = enc1[:length]
    return encoded, length


# |--------------------------------------|
# |       Training & Evaluation          |
# |--------------------------------------|
def train_model(model, train_dl, device, test_dl=None, epochs=10):
    # print("Training classifier...")
    start_t = time.time()
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    test_predictions = []
    test_ground_truths = []
    for epoch in range(epochs):
        model.train()
        epoch_t = time.time()
        sum_loss = 0.0
        total = 0
        correct = 0
        for x, y, l in train_dl:
            x = x.long()
            y = y.long()
            x, y = x.to(device), y.to(device)
            y_pred = model(x, l)
            optimizer.zero_grad()
            loss = F.cross_entropy(y_pred, y)
            preds = y_pred.argmax(dim=-1)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * y.shape[0]
            total += y.shape[0]

            correct += (preds == y).float().sum()
            # print(f"Epoch [{epoch}]: train loss {(sum_loss/total)}, train_acc: {correct/total}")
        results = evaluation(model, val_dl, device, test_dl)

        if len(results) == 2:  # both validation and testing took place
            val_loss = results[0][0]
            val_acc = results[0][1].item()
            test_loss = results[1][0]
            test_acc = results[1][1].item()
            predictions = results[1][2]
            ground_truths = results[1][3]
            test_predictions.extend(predictions)
            test_ground_truths.extend(ground_truths)
        else:
            val_loss = results[0]
            val_acc = results[1].item()
            test_loss = -1
            test_acc = -1

        # print("Epoch time; ", time.time() - epoch_t)
        print(f"Epoch [{epoch}]:"
              f" train loss {round(float(sum_loss / total), 4)},"
              f" train_acc: {round(float(correct / total), 4)}",
              f" | val loss {round(float(val_loss), 4)},"
              f"  val accuracy {round(float(val_acc), 4)}",
              f" | test loss {round(float(test_loss), 4)},"
              f" test accuracy {round(float(test_acc), 4)}",
              )

        train_accuracies.append(round(float(correct / total), 4))
        val_accuracies.append(round(float(val_acc), 4))
        test_accuracies.append(round(float(test_acc)))
        if len(val_accuracies) > 10 and val_accuracies[-1] < val_accuracies[-10]:
            print("Early stopping criterion has been reached.")
            break

    print("total time: ", time.time() - start_t)
    return train_accuracies, val_accuracies, test_accuracies, test_predictions, test_ground_truths


def evaluation(model, valid_dl, device, test_dl=None):
    model.eval()

    def run_evaluation(dataloader):
        correct = 0
        total = 0
        sum_loss = 0.0
        predictions = []
        ground_truths = []
        for x, y, l in dataloader:
            # evaluation data
            x = x.long()
            y = y.long()
            x, y = x.to(device), y.to(device)
            y_hat = model(x, l)
            loss = F.cross_entropy(y_hat, y)
            preds = y_hat.argmax(dim=-1)
            correct += (preds == y).float().sum()
            total += y.shape[0]
            sum_loss += loss.item() * y.shape[0]
            # confusion matrix data
            predictions.append(preds.item())
            ground_truths.append(y.item())
        return [sum_loss/total, correct/total, predictions, ground_truths]

    validation_results = run_evaluation(valid_dl)
    if test_dl:
        test_results = run_evaluation(test_dl)
        return validation_results, test_results
    return validation_results


tok = spacy.load('en_core_web_sm')
# |-----------------------------------------|
# |  OPTION 1: GENRE CLASSIFICATION DATASET |
# |-----------------------------------------|
# counts = trim_rare_words(movies_df, tok)
# word2idx = create_vocabulary(counts)
# movies_df['Plot_encoded'] = movies_df['Plot'].apply(lambda x: np.array(encode_sentence(x, tok, word2idx, PLOT_LENGTH)))
# num_classes = len(shortlisted_genres)
# title = 'Genre'
# print("number of classes %d" % num_classes)
# X = list(movies_df['Plot_encoded'])
# y = list(movies_df['genre_encoded'])
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# |-----------------------------------------|
# | OPTION 2: CAST CLASSIFICATION DATASET   |
# |-----------------------------------------|
X_train, y_train, = [], []
with open('./data/500/plots_train.csv', "r") as f:
    plot_reader = csv.reader(f, delimiter=',')
    plot_reader = list(iter(plot_reader))
    for row in plot_reader:
        y_train.append(int(row[0]))
        X_train.append(row[1])

X_valid, y_valid, = [], []
with open('./data/500/plots_valid.csv', "r") as f:
    plot_reader = csv.reader(f, delimiter=',')
    plot_reader = list(iter(plot_reader))
    for row in plot_reader:
        y_valid.append(int(row[0]))
        X_valid.append(row[1])

X_test, y_test, = [], []
with open('./data/500/plots_test.csv', "r") as f:
    plot_reader = csv.reader(f, delimiter=',')
    plot_reader = list(iter(plot_reader))
    for row in plot_reader:
        y_test.append(int(row[0]))
        X_test.append(row[1])

num_classes = 10
title = 'Cast'
assert num_classes == len(set(y_train)) == len(set(y_valid)) == len(set(y_test))
counts = trim_rare_words(X_train, tok)
word2idx = create_vocabulary(counts)
encode = lambda x: np.array(encode_sentence(x, tok, word2idx, PLOT_LENGTH))
X_train = [encode(x) for x in X_train]  # encoded
X_valid = [encode(x) for x in X_valid]  # encoded
X_test = [encode(x) for x in X_test]  # encoded


# ----------- Parameter sweep -----------
# batch_sizes = [16, 32, 64]
# embeddings = [50, 100, 200, 300]
# ----------- Test run --------------
batch_sizes = [32]
embeddings = [300]

for EMBEDDING_DIM in embeddings:
    HIDDEN_DIM = int(EMBEDDING_DIM // 2)
    for BATCH_SIZE in batch_sizes:
        print(f"\nEMBEDDING_DIM: [{EMBEDDING_DIM}]; HIDDEN_DIM: [{HIDDEN_DIM}]; BATCH_SIZE: [{BATCH_SIZE}]")

        # Set up Dataloaders
        train_ds = PlotsDataset(X_train, y_train)
        valid_ds = PlotsDataset(X_valid, y_valid)
        test_ds = PlotsDataset(X_test, y_test)
        train_dl = DataLoader(train_ds,
                              shuffle=True,
                              batch_size=BATCH_SIZE)
        val_dl = DataLoader(valid_ds)
        test_dl = DataLoader(test_ds)

        # Train model with Adam optimizer for 50 epochs
        model = TextClassifierLSTM(word2idx, EMBEDDING_DIM, HIDDEN_DIM, num_classes)
        parameters = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=5e-3, weight_decay=5e-4)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # print("running on ", device)
        model.to(device)
        train_accuracies, val_accuracies, test_accuracies, test_predictions, test_ground_truths = train_model(model, train_dl, device, epochs=50, test_dl=test_dl)

        # Plot confusion matrix if test dataset was used
        if len(test_predictions) > 1:
            conf_matrix = confusion_matrix(test_ground_truths, test_predictions)
            plt.imshow(conf_matrix, interpolation='None', cmap=plt.cm.Wistia)
            plt.title(f'{title} prediction Confusion Matrix - Test data')
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            ticks = list(range(num_classes))
            plt.yticks(ticks)
            plt.xticks(ticks)
            for i in range(num_classes):
                for j in range(num_classes):
                    plt.text(j-0.5, i, str(conf_matrix[i][j]))
            plt.show()

        # Write results
        with open(f"TRAIN_EMBEDDING_DIM:[{EMBEDDING_DIM}]_HIDDEN_DIM:[{HIDDEN_DIM}]_BATCH_SIZE:[{BATCH_SIZE}]", "w") as f:
            for value in train_accuracies:
                f.write(f"{str(value)}\n")
        with open(f"VALID_EMBEDDING_DIM:[{EMBEDDING_DIM}]_HIDDEN_DIM:[{HIDDEN_DIM}]_BATCH_SIZE:[{BATCH_SIZE}]", "w") as f:
            for value in val_accuracies:
                f.write(f"{str(value)}\n")
        with open(f"TEST_EMBEDDING_DIM:[{EMBEDDING_DIM}]_HIDDEN_DIM:[{HIDDEN_DIM}]_BATCH_SIZE:[{BATCH_SIZE}]", "w") as f:
            for value in test_accuracies:
                f.write(f"{str(value)}\n")

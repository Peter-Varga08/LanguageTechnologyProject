
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import SqueezeBertConfig, SqueezeBertForMultipleChoice, SqueezeBertTokenizer
import torch
import torch.nn as nn
import time
from sklearn.preprocessing import LabelEncoder

data_loc = "/data/s3861023/ltp_data"
movies_df = pd.read_csv(data_loc + "/wiki_movie_plots_deduped.csv")
movies_df = movies_df[(movies_df["Origin/Ethnicity"]=="American") | (movies_df["Origin/Ethnicity"]=="British")]
movies_df = movies_df[["Plot", "Genre"]]
drop_indices = movies_df[movies_df["Genre"] == "unknown" ].index
movies_df.drop(drop_indices, inplace=True)

# Combine genres: 1) "sci-fi" with "science fiction" &  2) "romantic comedy" with "romance"
movies_df["Genre"].replace({"sci-fi": "science fiction", "romantic comedy": "romance"}, inplace=True)

# Choosing movie genres based on their frequency
shortlisted_genres = movies_df["Genre"].value_counts().reset_index(name="count").query("count > 200")["index"].tolist()
movies_df = movies_df[movies_df["Genre"].isin(shortlisted_genres)].reset_index(drop=True)

# Shuffle DataFrame
movies_df = movies_df.sample(frac=1).reset_index(drop=True)

# Sample roughly equal number of movie plots from different genres (to reduce class imbalance issues)
movies_df = movies_df.groupby("Genre").head(100).reset_index(drop=True)

label_encoder = LabelEncoder()
movies_df["genre_encoded"] = label_encoder.fit_transform(movies_df["Genre"].tolist())

movies_df = movies_df[["Plot", "Genre", "genre_encoded"]]
train_df, eval_df = train_test_split(movies_df, test_size=0.2, stratify=movies_df["Genre"], random_state=42)

class movie_dataset(Dataset):
    def __init__(self, tokenizer):
        super().__init__()



        label_encoder = LabelEncoder()
        movies_df = pd.read_csv(data_loc + "/ltp_data/wiki_movie_plots_deduped.csv")
        movies_df = movies_df[(movies_df["Origin/Ethnicity"]=="American") | (movies_df["Origin/Ethnicity"]=="British")]
        movies_df = movies_df[["Plot", "Genre"]]
        drop_indices = movies_df[movies_df["Genre"] == "unknown" ].index
        movies_df.drop(drop_indices, inplace=True)

        # Combine genres: 1) "sci-fi" with "science fiction" &  2) "romantic comedy" with "romance"
        movies_df["Genre"].replace({"sci-fi": "science fiction", "romantic comedy": "romance"}, inplace=True)

        # Choosing movie genres based on their frequency
        shortlisted_genres = movies_df["Genre"].value_counts().reset_index(name="count").query("count > 200")["index"].tolist()
        movies_df = movies_df[movies_df["Genre"].isin(shortlisted_genres)].reset_index(drop=True)
        movies_df = movies_df.groupby("Genre").head(100).reset_index(drop=True)

        data = []
        labels = []
        data_sepperate = movies_df["Plot"]
        for i, line in enumerate(data_sepperate):
            print(i)
            tokens = tokenizer.tokenize(line[:512])
            data_idxs = tokenizer.encode(line[:512])
            if(len(data_idxs) > 512):
                data_idxs = data_idxs[:512]
            data.append(data_idxs)
        #data = data.values()
        labels = label_encoder.fit_transform(movies_df["Genre"])#needs encoding
        #labels = labels.values()
        self.data = data
        self.labels = labels
        print(labels)
        print(labels[1])

    def __getitem__(self, index):
        return torch.Tensor(self.data[index]).long(), torch.Tensor(self.labels[index]).int()

    def __len__(self):
        return len(self.data)




#train_features, train_labels = next(iter(train_dataloader))
#print("\nCan't iterate \n")
#print(f"Feature batch shape: {train_features.size()}")
#print(f"Labels batch shape: {train_labels.size()}")
#plot = train_features[0]
#label = train_labels[0]
#print(f'plot: {plot}')
#print(f"Label: {label}")
def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        for batch in loader:
            data, labels = batch
            out = model(input_ids=data, labels=labels)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.numel()



    print(correct, total)
    model.train()
    return correct/total

def train(model, train_loader, valid_loader, test_loader, epochs):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0.0
        total_tokens = 0.0
        for i, batch in enumerate(train_loader):

            if i % 5 == 0 and i > 0:
                tps = total_tokens/ (time.time()-start_time)
                print("[Epoch %d, Iter %d] loss: %.4f, toks/sec: %d" % (epoch, i, total_loss/5, tps))
                start_time = time.time()
                total_loss = 0.0
                total_tokens = 0.0

            data, labels = batch
            optimizer.zero_grad()
            out = model(input_ids=data, labels=labels)
            out.loss.backward()
            optimizer.step()
            total_loss += out.loss.item()
            total_tokens += data.numel()

        acc = evaluate(model, valid_loader)
        print("[Epoch %d] Acc (valid): %.4f" % (epoch, acc))
        acc = evaluate(model, test_loader)
        print("[Epoch %d] Acc (test): %.4f" % (epoch, acc))
        start_time = time.time() # so tps stays consistent

    print("############## END OF TRAINING ##############")
    acc = evaluate(model, valid_loader)
    print("Final Acc (valid): %.4f" % (acc))
    acc = evaluate(model, test_loader)
    print("Final Acc (test): %.4f" % (acc))
def padding_collate_fn(batch):
    """ Pads data with zeros to size of longest sentence in batch. """
    data, labels = zip(*batch)
    largest_sample = max([len(d) for d in data])
    padded_data = torch.zeros((len(data), largest_sample), dtype=torch.long)
    padded_labels = labels
    for i, sample in enumerate(data):
        padded_data[i, :len(sample)] = sample

    return padded_data, padded_labels

# pretrained = 'squeezebert/squeezebert-uncased'
# tokenizer = SqueezeBertTokenizer.from_pretrained(pretrained)
# tokenizer.do_basic_tokenize = False
# train_dataset = movie_dataset(tokenizer)
# train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True,collate_fn=padding_collate_fn,)
# configuration = SqueezeBertConfig.from_pretrained(pretrained)
# configuration.num_labels = 1#movies_df['Genre'].nunique()
# configuration.num_hidden_layers = 1
# configuration.num_attention_heads = 1
# configuration.output_attentions = True
# model = SqueezeBertForMultipleChoice(configuration)
# train(model, train_dataloader, train_dataloader, train_dataloader, 16)

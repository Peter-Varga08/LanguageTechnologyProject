import pandas as pd
from pandas.core.reshape.reshape import get_dummies
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
import time
from torch.utils.data import DataLoader 
from transformers import SqueezeBertConfig, SqueezeBertTokenizer, SqueezeBertForTokenClassification, BertForTokenClassification, BertTokenizer,SqueezeBertForSequenceClassification

data = pd.read_csv('data/imdb_5000/tmdb_5000_movies.csv')

print('Average number of characters per plor/overview: %f' % np.mean(data['overview'].astype(str).map(len)))
print('Number of movies in the dataset %d' % len(data))
# print(list(data.columns))
print('\nExample \n %s \n\n' % data['overview'][np.random.randint(len(data))])

multi_genre = False

if multi_genre: 
    # Extract data from multiple genres per movie
    genre_title_data = []
    genre_labels = pd.DataFrame(columns=['id', 'name'])
    for idx, row in data.iterrows():
        genres = []
        convertedDict = json.loads(row['genres'])
        if convertedDict == []:
            continue
        for genre in convertedDict:
            genres.append(genre['id'])

        # Extract labels 
        # toDataFrame = pd.DataFrame.from_dict(convertedDict)
        # genre_labels = pd.concat([genre_labels,toDataFrame],ignore_index=True)

        genre_title_data.append({'title_id': row['id'], 'genres': genres})


    genre_title_data = pd.DataFrame.from_dict(genre_title_data)
    genre_labels = genre_labels.drop_duplicates()
    print(genre_title_data.head)
    print(genre_labels)

    # genre_labels.to_csv(r'./data/imdb_5000/genre_labels.csv', index = False, header=True)
# else:
#     genre_title_data = []
#     genre_data =[]
#     title_data = []
#     genre_overview_data = []

#     for idx, row in data.iterrows():
#         convertedDict = json.loads(row['genres'])
#         # Skip if there is no genre information
#         if convertedDict == []:
#             continue
#         for genre in convertedDict:
#             # genres.append(genre['id'])
#             genre_data.append(genre['id'])
#             title_data.append(row['id'])
#             genre_overview_data.append({'overview': row['overview'], 'genre': genre['id']})
#             genre_title_data.append({'title_id': row['id'], 'genre': genre['id']})

#         # genre_title_data.append({'title_id': row['id'], 'genres': genres})

#     genre_title_data = pd.DataFrame.from_dict(genre_title_data)
    print(genre_title_data.head)

class genreDataset:
    def __init__(self, dataset, tokenizer):
        self.data = []
        self.labels = []

        for idx, row in dataset.iterrows():
            convertedDict = json.loads(row['genres'])
            # Skip if there is no genre information
            if convertedDict == []:
                continue
            for genre in convertedDict:
                try:
                    # tokens = tokenizer.tokenize(row['overview'])
                    data_idxs = tokenizer.encode(row['overview'])
                    self.data.append(data_idxs)
                    self.labels.append(genre['id'])
                except:
                    print('No overview found')

# print(genre_overview_data[0:10])

labels = pd.read_csv('data/imdb_5000/genre_labels.csv')
print(len(labels))
pretrained = 'squeezebert/squeezebert-uncased'
tokenizer = SqueezeBertTokenizer.from_pretrained(pretrained)
tokenizer.do_basic_tokenize = False

genreData = genreDataset(data, tokenizer)

train_data = genreData.data[0:int(len(genreData.data)*0.7)]
train_labels = genreData.labels[0:int(len(genreData.labels)*0.7)]

test_data = genreData.data[int(len(genreData.data)*0.7):]
test_labels = genreData.labels[int(len(genreData.labels)*0.7):]

train_loader = DataLoader(train_data,
    shuffle=True,
    batch_size=4)

test_loader = DataLoader(test_data,
    batch_size=4)


print(genreData.labels[0:10])
print(genreData.data[0:10])
# data = 

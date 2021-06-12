# import torch
# import torch.nn as nn
import pandas as pd
# import numpy as np
# import re
# import spacy
# import torch.nn.functional as F
# import string
# import csv
import os
# from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from sklearn.metrics import mean_squared_error
# from collections import Counter
# from torch.utils.data import DataLoader, Dataset
# from sklearn.model_selection import train_test_split

# # from dataloader_movies import movies_df, label_encoder, shortlisted_genres
# from annoy import AnnoyIndex
# from spacy.lang.en.examples import sentences

# tok = spacy.load('en_core_web_sm')
# print(os.listdir('../../../../data/s3861023/ltp_data/wiki_movie_plots_deduped.csv'))
movies_df = pd.read_csv("data/s3861023/ltp_data/wiki_movie_plots_deduped.csv")
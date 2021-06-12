import numpy as np
import pandas as pd
import os, json, gc, re, random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import torch, transformers, tokenizers

movies_df = pd.read_csv("wiki_movie_plots_deduped.csv")


movies_df = movies_df[(movies_df["Origin/Ethnicity"]=="American") | (movies_df["Origin/Ethnicity"]=="British")]
movies_df = movies_df[["Plot", "Genre"]]
drop_indices = movies_df[movies_df["Genre"] == "unknown" ].index
movies_df.drop(drop_indices, inplace=True)#

# Combine genres: 1) "sci-fi" with "science fiction" &  2) "romantic comedy" with "romance"
movies_df["Genre"].replace({"sci-fi": "science fiction", "romantic comedy": "romance"}, inplace=True)

# Choosing movie genres based on their frequency
shortlisted_genres = movies_df["Genre"].value_counts().reset_index(name="count").query("count > 200")["index"].tolist()
movies_df = movies_df[movies_df["Genre"].isin(shortlisted_genres)].reset_index(drop=True)

# Shuffle DataFrame
movies_df = movies_df.sample(frac=1).reset_index(drop=True)

# Sample roughly equal number of movie plots from different genres (to reduce class imbalance issues)
movies_df = movies_df.groupby("Genre").head(400).reset_index(drop=True)

label_encoder = LabelEncoder()
movies_df["genre_encoded"] = label_encoder.fit_transform(movies_df["Genre"].tolist())

movies_df = movies_df[["Plot", "Genre", "genre_encoded"]]

#movies_df.to_csv("Wiki_movie_plots.csv")


from simpletransformers.classification import ClassificationModel

model_args = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "save_model_every_epoch": False,
    "save_eval_checkpoints": False,
    "max_seq_length": 512,
    "train_batch_size": 1,
    "num_train_epochs": 20,
}

# Create a ClassificationModel
#model = ClassificationModel('distilbert', 'distilbert-base-uncased', num_labels=len(shortlisted_genres), args=model_args) #{'mcc': 0.4534451424142884, 'eval_loss': 7.120329482118355}
model = ClassificationModel('xlnet', 'xlnet-large-cased', num_labels=len(shortlisted_genres), args=model_args) #{'mcc': 0.4534451424142884, 'eval_loss': 7.120329482118355}



train_df, eval_df = train_test_split(movies_df, test_size=0.2, stratify=movies_df["Genre"], random_state=42)

# Train the model
model.train_model(train_df[["Plot", "genre_encoded"]])

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df[["Plot", "genre_encoded"]])
print(result)




predicted_genres_encoded = list(map(lambda x: np.argmax(x), model_outputs))
predicted_genres = list(label_encoder.inverse_transform(predicted_genres_encoded))
eval_gt_labels = eval_df["Genre"].tolist()
class_labels = list(label_encoder.classes_)

plt.figure(figsize=(22,18))
cf_matrix = confusion_matrix(predicted_genres, eval_gt_labels, class_labels)
ax = sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, cmap="YlGnBu")
ax.set_xlabel('Predicted Genres', fontsize=20)
ax.set_ylabel('True Genres', fontsize=20)
ax.set_title('Confusion Matrix', fontsize=20)
ax.set_xticklabels(class_labels, rotation=90, fontsize=18)
ax.set_yticklabels(class_labels, rotation=0, fontsize=18)

plt.savefig("confusion_matrix.png")



#for _ in range(5):

#    random_idx = random.randint(0, len(eval_df)-1)
#    text = eval_df.iloc[random_idx]['Plot']
#    true_genre = eval_df.iloc[random_idx]['Genre']

    # Predict with trained multiclass classification model
#    predicted_genre_encoded, raw_outputs = model.predict([text])
#    predicted_genre_encoded = np.array(predicted_genre_encoded)
#    predicted_genre = label_encoder.inverse_transform(predicted_genre_encoded)[0]

#    print(f'\nTrue Genre:'.ljust(16,' '), f'{true_genre}\n')
#    print(f'Predicted Genre: {predicted_genre}\n')
#    print(f'Plot: {text}\n')
#    print("-------------------------------------------")

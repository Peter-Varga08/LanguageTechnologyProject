import pandas as pd
import re
import unicodedata


def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

data = pd.read_csv('data/wiki_movie_plots_deduped.csv')
print("Amount of movies in entire dataset:", len(data))

# 0) Filter out movies with no cast
data = data[(data['Cast'].isna() == False) & (data['Cast'] != 'Unknown') & (data['Cast'] != '')]
print("Amount of movies in dataset without 'NaN' and 'Unknown' cast:", len(data))

# 1) Count movies per person
cast_per_movie = list(data["Cast"])
# pattern = re.compile("([a-zA-Z]+\.?)(\s[a-zA-Z]+\.?)*")
# matches = re.findall(pattern, cast_per_movie[1])
cast_per_movie_split = [cast.split(',') for cast in cast_per_movie]
total_names = {}
for cast in cast_per_movie_split:
    for name in cast:
        name = name.strip()
        names = None
        if 'and' in name:
            names = name.split('and')
        elif '&' in name:
            names = name.split('&')
        elif '\r\n' in name:
            names = name.split('\r\n')
        if names:  # e.g. 'Marlon Brando and Charlie Chaplin' or 'Marlon Brando & Charlie Chaplin'
            for name in names:
                name = name.strip()
                if name not in total_names.keys():
                    total_names[name] = 1
                else:
                    total_names[name] += 1
        else:
            if name not in total_names.keys():
                total_names[name] = 1
            else:
                total_names[name] += 1

# Filter out invalid names, e.g. single letters or empty strings due to wrongly formatted data
total_names = {name: count for name, count in total_names.items() if len(name.split()) > 1}
total_names_list = list(total_names.keys())

# Sort persons based on movie count
print("Sorting names based on movie appearances...")
sorted_names = {}
sorted_keys = sorted(total_names, key=total_names.get, reverse=True)
for key in sorted_keys:
    sorted_names[key] = total_names[key]

# Sanity check: no invalid name should occur
print("Top 15 names:", list(sorted_names.keys())[0:15])

# Get 100 labels by retrieving the top 100
labels = list(sorted_names.keys())[0:100]

# Get movie plots that correspond to an actor/actress from top1-100
plots = []
for label in labels:
    curr_label = []
    for i in range(1, len(data)):
        if label in data.iloc[i]['Cast']:
            curr_label.append(data.iloc[i]['Plot'])
    plots.append(curr_label)






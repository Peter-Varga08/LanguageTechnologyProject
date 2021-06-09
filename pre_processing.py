import pandas as pd
from sklearn.model_selection import train_test_split


def split_names(name):
    names = None
    if ' and ' in name:
        names = name.split(' and ')
    elif ' & ' in name:
        names = name.split(' & ')
    elif '\r\n' in name:
        names = name.split('\r\n')
    return names

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
        names = split_names(name)
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
total_names = {name: count for name, count in total_names.items() if len(name.strip()) > 3}
total_names_list = list(total_names.keys())

# Sort persons based on movie count
print("Sorting names based on movie appearances...")
sorted_names = {}
sorted_keys = sorted(total_names, key=total_names.get, reverse=True)
for key in sorted_keys:
    sorted_names[key] = total_names[key]

# Sanity check: no invalid name should occur
print("Top 15 names:", list(sorted_names.keys())[0:15])

# Get 10 labels by retrieving the top 10
labels = list(sorted_names.keys())[0:10]

# Get movie plots that correspond to an actor/actress from top10
plots = {name: [] for name in total_names_list}
for i in range(len(data)):
    cast = data.iloc[i]['Cast']
    cast = cast.split(',')
    for name in cast:
        name = name.strip()
        names = split_names(name)
        if names:
            for name in names:
                name = name.strip()
                if len(name) > 3:
                    plots[name].append(data.iloc[i]['Plot'])
        else:
            if len(name) > 3:
                plots[name].append(data.iloc[i]['Plot'])

# Filter plots shorter than 1000 characters
MIN_LENGTH = 1000
plot_lengths = {name: [len(plot) for plot in plots[name]] for name in labels}
max_length = 0
for name, lengths in plot_lengths.items():
    for length in lengths:
        if length > max_length:
            max_length = length

# Filtering and Augmenting data:
# - Discard each plot shorther than MIN_LENGTH
# - Split up the plots to subplots based on MIN_LENGTH
plots_filtered = {label: [] for label in labels}
for name in labels:
    for idx, plot in enumerate(plots[name]):
        if plot_lengths[name][idx] > MIN_LENGTH:
            for j in range(max_length//MIN_LENGTH):
                if len(plot[0 + j*1000: 1000*(j+1)]) == MIN_LENGTH:
                    plots_filtered[name].append(plot[0 + j*1000: 1000*(j+1)])
                # else:
                #     plots_filtered[name].append(plot[0 + j*1000:])
                #     break


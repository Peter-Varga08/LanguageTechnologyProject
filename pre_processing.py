import csv
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

# 2) Filter out invalid names, e.g. single letters or empty strings due to wrongly formatted original data
total_names = {name: count for name, count in total_names.items() if len(name.strip()) > 3}
total_names_list = list(total_names.keys())

# 3) Sort persons based on movie count
print("Sorting names based on movie appearances...")
sorted_names = {}
sorted_keys = sorted(total_names, key=total_names.get, reverse=True)
for key in sorted_keys:
    sorted_names[key] = total_names[key]

# Sanity check: no invalid name should occur
print("Top 15 names:", list(sorted_names.keys())[0:15])

# 4) Get all movie plots
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

# 5) Get 10 labels by retrieving the top 10
labels = list(sorted_names.keys())[0:10]
# 6) Get max plot length, subsequently used for filtering
plot_lengths = {name: [len(plot) for plot in plots[name]] for name in labels}
max_length = 0
for name, lengths in plot_lengths.items():
    for length in lengths:
        if length > max_length:
            max_length = length

# 7) Filtering and Augmenting data:
# - Discard each plot shorter than MIN_LENGTH
# - Split up the plots to subplots based on MIN_LENGTH
MIN_LENGTH = 1000
plots_filtered = {label: [] for label in labels}
for name in labels:
    for idx, plot in enumerate(plots[name]):
        if plot_lengths[name][idx] > MIN_LENGTH:
            for j in range(max_length//MIN_LENGTH):  # number of slices available within current plot
                if len(plot[0 + j*MIN_LENGTH: MIN_LENGTH*(j+1)]) == MIN_LENGTH:
                    plots_filtered[name].append(plot[0 + j*MIN_LENGTH: MIN_LENGTH*(j+1)])
                # else:
                #     plots_filtered[name].append(plot[0 + j*1000:])
                #     break


# 8) Train/Valid/Test split
X = []
y = []
for idx, name in enumerate(plots_filtered):
    for plot in plots_filtered[name]:
        X.append(plot)
        y.append(idx)

# 8/A) Split data to train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 8/B) Split data to train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

with open('plots_train.csv', 'w') as f:
    train_writer = csv.writer(f, delimiter=',')
    for x, y in zip(X_train, y_train):
        train_writer.writerow([y, x])

with open('plots_valid.csv', 'w') as f:
    train_writer = csv.writer(f, delimiter=',')
    for x, y in zip(X_val, y_val):
        train_writer.writerow([y, x])

with open('plots_test.csv', 'w') as f:
    train_writer = csv.writer(f, delimiter=',')
    for x, y in zip(X_test, y_test):
        train_writer.writerow([y, x])

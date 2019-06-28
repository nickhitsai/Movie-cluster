import pandas as pd
import json

from model import MyModel

def clean_data_for_title(x):
    if isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''

def clean_data_for_genres(x):
    if isinstance(x, str):
        return str.lower(x.replace("\'", "\""))
    else:
        return ''

def clean_data_for_overview(x):
    if isinstance(x, str):
        x = ' '.join(filter(str.isalpha, i) for i in x.split(' '))
        return str.lower(x)
    else:
        return ''

# 
# Preprocessing the input data to data that can be used to train a model.
# such as remove duplcate items, remove empty items, and so on
# Keep the input data to the original shape
# 
def proprocessing():
    used_col = ['original_title', 'genres', 'overview']
    df = pd.read_csv('movies_metadata.csv', usecols=used_col)
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows if one of the column in the row is empty
    for col in used_col:
        df = df[pd.notnull(df[col])]

    # Apply some specific operation to the Series
    # Please refer to the function name
    df['original_title'] = df['original_title'].apply(clean_data_for_title)
    df['genres'] = df['genres'].apply(clean_data_for_genres)
    # Transform the json to list
    df['genres'] = df['genres'].apply(json.loads)
    df['overview'] = df['overview'].apply(clean_data_for_overview)

    return df

# 
# K-fold
# eval to find best model
# 
def train():
    pass

def main():
    # 1. preprocess the input data
    # 2. train model
    #   2.1 customize as the same interface as scikit-learn model
    #   2.2 K-fold
    #   2.3 Evaluation
    # 3. save the best model for predicting
    pass

if __name__ == "__main__":
    main()
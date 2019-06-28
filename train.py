import pandas as pd
import numpy as np
import json, joblib

from model import MyModel
from sklearn.model_selection import KFold

used_col = ['original_title', 'genres', 'overview']
model_name = 'mymodel.joblib'
data_used_for_training = 10000
data_used_for_testing = 1000

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
        x = ' '.join(''.join(char for char in word if char.isalpha()) for word in x.split(' '))
        return str.lower(x)
    else:
        return ''

# 
# Preprocessing the input data to data that can be used to train a model.
# such as remove duplcate items, remove empty items, and so on
# Keep the input data to the original shape
# 
def preprocessing():
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
def train(input_dataframe):
    fold_num = 5
    kf = KFold(n_splits=fold_num)
    best_score = 0
    best_model = None
    history_score = []
    cnt = 1

    for train_idx, test_idx in kf.split(input_dataframe):
        train = input_dataframe.iloc[train_idx[:data_used_for_training]]
        test = input_dataframe.iloc[test_idx[:data_used_for_testing]]
        model = MyModel()
        X = train.loc[:, 'overview']
        y = train.loc[:, 'genres']
        model.fit(X, y)

        truth = test.loc[:, 'genres']
        prediction = model.predict(test.loc[:, 'overview'])

        score = evaluation(truth, prediction)

        history_score.append(score)
        if score > best_score:
            best_score = score
            best_model = model
        
        print('Accuracy of fold %d: %.2f' % (cnt, score))
        cnt += 1
    
    print('Best score: ' + str(best_score))
    print('Worst score: ' + str(min(history_score)))
    print('Average score: ' + str(np.array(history_score).mean()))
    best_model.save_weights(model_name)

def evaluation(ground_truth, prediction):
    cnt = 0
    for i in range(len(prediction)):
        truth_list = [item['name'] for item in ground_truth.iloc[i]]
        if prediction[i] in truth_list:
            cnt += 1

    return cnt/len(prediction) if len(prediction) != 0 else 0
        

def main():
    # 1. preprocess the input data
    # 2. train model
    #   2.1 customize as the same interface as scikit-learn model
    #   2.2 K-fold
    #   2.3 Evaluation
    # 3. save the best model for predicting
    train(preprocessing())

if __name__ == "__main__":
    main()
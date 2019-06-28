import pandas as pd
import numpy as np
import json, joblib, os

from model import MyModel
from sklearn.model_selection import KFold

# Only use these column for this model
used_col = ['original_title', 'genres', 'overview']
# Model path
model_name = 'mymodel.joblib'
input_dataset = 'movies_metadata.csv'

# Please tune these parameters based on your computing power
data_used_for_training = 1000
data_used_for_testing = 100

# Since the input title would not contain any space, remove it from the dataset.
# Transform to lower case
def clean_data_for_title(x):
    if isinstance(x, str):
        return str.lower(x.replace(" ", ""))
    else:
        return ''

# It's for json library.
# It requires double quote.
# Transform to lower case
def clean_data_for_genres(x):
    if isinstance(x, str):
        return str.lower(x.replace("\'", "\""))
    else:
        return ''

# Keep the space and letter in the paragraph.
# Also, transform it to lower case.
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
    df = pd.read_csv(input_dataset, usecols=used_col)
    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove rows if one of the column in the row is empty
    for col in used_col:
        df = df[pd.notnull(df[col])]

    # Apply some specific operation to the Series
    # Please refer to the function name
    df['original_title'] = df['original_title'].apply(clean_data_for_title)
    df['genres'] = df['genres'].apply(clean_data_for_genres)
    df['overview'] = df['overview'].apply(clean_data_for_overview)

    # Transform the json to list
    df['genres'] = df['genres'].apply(json.loads)

    return df

# 
# K-fold
# eval to find best model
# 
def train(input_dataframe):
    # Split the dataset to train and test
    # Since the fold_num is 5, every train data would be around 80% of the dataset and test data would be around 20% of the dataset.
    fold_num = 5
    kf = KFold(n_splits=fold_num)

    # Store the best model for prediction
    best_score = 0
    best_model = None
    history_score = []

    # This is used for showing the current fold number
    cnt = 1

    # Training part
    for train_idx, test_idx in kf.split(input_dataframe):
        # Split the training and testing part from input dataframe.
        # It is based on the index
        train = input_dataframe.iloc[train_idx[:data_used_for_training]]
        test = input_dataframe.iloc[test_idx[:data_used_for_testing]]

        # Init the model class
        model = MyModel()

        # Prepare the data for training the model
        X = train.loc[:, 'overview']
        y = train.loc[:, 'genres']

        # Train the model
        model.fit(X, y)

        # Prepare the ground truth and prediction for evaluating the performance.
        truth = test.loc[:, 'genres']
        prediction = model.predict(test.loc[:, 'overview'])

        # Compute the score
        score = evaluation(truth, prediction)

        # Store all the score in this list
        history_score.append(score)

        # Store the best model and score
        if score > best_score:
            best_score = score
            best_model = model
        
        # Print the current states
        print('Accuracy of fold %d: %.2f' % (cnt, score))
        cnt += 1
    
    # Print the contents
    print('Best score: ' + str(best_score))
    print('Worst score: ' + str(min(history_score)))
    print('Average score: ' + str(np.array(history_score).mean()))

    # Save the best model
    best_model.save_weights(model_name)

# 
# Evaluate the model
# 
def evaluation(ground_truth, prediction):
    # Used for summing up the truth positive
    cnt = 0

    # Iterate rows in prediction
    for i in range(len(prediction)):
        # Since the genres in dataset would contain multiple values, it retrives the value from the genres column.
        # Example:
        # in: [{'id': 16, 'name': 'Animation'}, {'id': 35, 'name': 'Comedy'}, {'id': 10751, 'name': 'Family'}]
        # out: ['Animation', 'Comedy', 'Family']
        truth_list = [item['name'] for item in ground_truth.iloc[i]]

        # If the prediction is one of the truth, it is true positive.
        if prediction[i] in truth_list:
            cnt += 1

    # Prevent from using empty input
    return cnt/len(prediction) if len(prediction) != 0 else 0
        

def main():
    # Check the input dataset exists.
    if not os.path.isfile(input_dataset):
        print('Please download the input dataset(movies_metadata.csv) from the following link:')
        print('https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv')
        exit(0)

    # 1. preprocess the input data
    # 2. train model
    #   2.1 customize as the same interface as scikit-learn model
    #   2.2 K-fold
    #   2.3 Evaluation
    # 3. save the best model for predicting
    train(preprocessing())

if __name__ == "__main__":
    main()
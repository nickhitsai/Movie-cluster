import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# model class
#     fit
#     fit_trainsform
#     predict
#     save_weights
#     load_weights

class MyModel(object):
    def __init__(self):
        
        self._model = TfidfVectorizer(stop_words='english')

        # init some variables
        self._tfidf_matrix = None
        self._model_path = None
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._tfidf_matrix = self._model.fit_transform(X)
    
    def fit_transform(self, X, y):
        return self.fit(X, y)

    def predict(self, X):
        # Construct pairwise cosine similarity from prediction data to dataset
        consine_similarity = linear_kernel(self._model.transform(X), self._tfidf_matrix)
        sim_scores = list(list(enumerate(x)) for x in consine_similarity)

        # Sort the value by score.
        # Only get the top 1000 similar movies.
        sim_scores = [sorted(sim_score, key=lambda x:x[1], reverse=True)[:1000] for sim_score in sim_scores]

        # Get the indices from sorted lists
        sim_indices = [[i[0] for i in sim_score] for sim_score in sim_scores]

        # Get the related row from dataset
        related_movie_genres_list = [self._y.iloc[sim_indice].tolist() for sim_indice in sim_indices]

        # Prepare the result
        res = []

        # Iterate through the output
        for related_movie_genres in related_movie_genres_list:
            # Trying to find genre which appers the most frequently in these related movies
            counting = {}
            # Iterate all movie
            for movie_generes in related_movie_genres:
                # Iterage all genres
                for generes in movie_generes:
                    # Count the frequency
                    counting[generes['name']] = counting[generes['name']] + 1 if generes['name'] in counting else 1

            # Convert the dictionary to list
            cnt_list = [(k, v) for k, v in counting.items()]
            # Use the genre which appears the most frequently
            res.append(sorted(cnt_list, key=lambda x:x[1], reverse=True)[0][0])

        return np.array(res)

    def save_weights(self, filepath):
        try:
            joblib.dump((self._model, self._tfidf_matrix, self._y), filepath)
        except e:
            print(e)
            print('Model does not dump!')

    def load_weights(self, filepath):
        try:
            (self._model, self._tfidf_matrix, self._y) = joblib.load(filepath)
        except e:
            print(e)
            print('Model does not load!')
        
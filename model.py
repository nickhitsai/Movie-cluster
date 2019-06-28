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
        self._model_path = None
        self._consine_similarity = None
        self._tfidf_matrix = None
        self._X = None
        self._y = None

    def fit(self, X, y):
        self._X = X
        self._y = y
        self._tfidf_matrix = self._model.fit_transform(X)
    
    def fit_transform(self, X, y):
        return self.fit(X, y)

    def predict(self, X):
        consine_similarity = linear_kernel(self._model.transform(X), self._tfidf_matrix)
        sim_scores = list(list(enumerate(x)) for x in consine_similarity)
        sim_scores = [sorted(sim_score, key=lambda x:x[1], reverse=True)[:1000] for sim_score in sim_scores]
        sim_indices = [[i[0] for i in sim_score] for sim_score in sim_scores]

        related_movie_genres_list = [self._y.iloc[sim_indice].tolist() for sim_indice in sim_indices]
        res = []
        for related_movie_genres in related_movie_genres_list:
            counting = {}
            for movie_generes in related_movie_genres:
                for generes in movie_generes:
                    counting[generes['name']] = counting[generes['name']] + 1 if generes['name'] in counting else 1

            cnt_list = [(k, v) for k, v in counting.items()]
            res.append(sorted(cnt_list, key=lambda x:x[1], reverse=True)[0][0])

        # res = []
        # for item in X:
        #     sim_scores = list(enumerate(linear_kernel()))

        #     # Get the pairwsie similarity scores of all movies with that movie
        #     sim_scores = list(enumerate(self._consine_similarity[idx]))

        #     # Sort the movies based on the similarity scores
        #     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        #     # Get the scores of the 100 most similar movies
        #     sim_scores = sim_scores[:100]

        #     # Get the movie indices
        #     movie_indices = [i[0] for i in sim_scores]

        #     # Return the top 10 most similar movies
        #     return df2['title'].iloc[movie_indices]

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
        
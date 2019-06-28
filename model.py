# model class
#     fit
#     fit_trainsform
#     predict
#     save_weights
#     load_weights

class MyModel(object):
    def __init__(self):
        self._model = None
        self._model_path = None

    def fit(self, X, y):
        pass
    
    def fit_transform(self, X, y):
        return self.fit(X, y)

    def predict(self, X):
        pass

    def save_weights(self, argPath):
        pass

    def load_weights(self, argPath):
        pass
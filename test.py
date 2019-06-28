import argparse, joblib, json
import numpy as np

from train import model_name, clean_data_for_overview
from model import MyModel

# Get input from command line interface
# format: movie_classifier --title <title> --description <description>
# output format:
# {
#     "title": "Othello",
#     "description": "The evil Iago pretends to be friend of Othello in order to manipulate him to serve his own end in the film version of this Shakespeare classic.",
#     "genre": "Drama"
# }

# 
# Parse parameter lists
# load model
# predict
# 
def predict(title, description):
    model = MyModel()
    model.load_weights(model_name)
    des = clean_data_for_overview(description)
    genre = model.predict(np.array([des]))
    res = r'''{
        "title": "%s",
        "description": "%s",
        "genre": "%s"
}''' % (title, description, genre[0])
    print(res)

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--title', type=str, help='movie title')
    parser.add_argument('--description', type=str, help='movie description')

    args = parser.parse_args()
    title = args.title
    description = args.description

    predict(title, description)

if __name__ == "__main__":
    main()
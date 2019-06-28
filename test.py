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
# Check the input data
# Only take the letter into account. 
# There must be at least one letter in the input.
def input_check(input_string):
    tmp = ''.join(in_str for in_str in input_string if in_str.isalpha())
    return len(tmp) == 0

# 
# Parse parameter lists
# load model
# predict
# 
def predict(title, description):
    # Load model and weights
    model = MyModel()
    model.load_weights(model_name)

    # Clean the description
    des = clean_data_for_overview(description)

    # Predict
    genre = model.predict(np.array([des]))

    # Prepare the output
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

    if title is None or input_check(title):
        print('Please enter non-empty movie title and try again')
        return
    
    if description is None or input_check(description):
        print('Please enter non-empty movie description and try again')
        return

    predict(title, description)

if __name__ == "__main__":
    main()
# Machine learning practice
## This repo is related to movie clustring. Reference: Kaggle's [the movie dataset](https://www.kaggle.com/rounakbanik/the-movies-dataset/version/7#movies_metadata.csv)

# How to run it
## Environment
The environment on my Macbook Air(2014) is listed as follow:

* Python 3.5
* NumPy==1.16.4
* pandas==0.24.2
* scikit-learn==0.21.2
* joblib==0.13.2

There are also some dependency. The pip would help you to install them all.
There is also one requirements.txt for reference.

## Training part
Once the libraries are all installed. 

Use the following command to train the model:
`python train.py`

It brings you the most basic model.
Please change the variables on top of the `train.py`, named 'data_used_for_training' and 'data_used_for_testing' based on your computing power.

## Testing part
After training the model, there is one file, called 'mymodel.joblib', which would be generated in the same folder.
It must exist to proceed the following steps.

Use the following command to run the `test.py`
`python test.py --title "Othello" --description "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstan..."`

Sample output:
```
{
        "title": "Othello",
        "description": "Led by Woody, Andy's toys live happily in his room until Andy's birthday brings Buzz Lightyear onto the scene. Afraid of losing his place in Andy's heart, Woody plots against Buzz. But when circumstan...",
        "genre": "drama"
}
```

Feel free to try another kinds of title and dexcription.

# The algorithm for this clustring job
The idea is that `I would like to use some words to identify the category of the movie`.
Example: if `happy` and `laugh` exist in the decription, there is a good chance that it belongs to `comedy`.

In the begining, I try to build up a relation between the keyword and the genres.
However, I just don't know how to decide the weight of every distinct words.
Also, there are lots of useless words, such as 'the', 'a', 'if', and so on.
These words happends a lot. I don't know how to remove these words in a proper way.

After some research, I found the `TfidfVectorizer` in scikit-learn.
I found it because I was trying to figure out how to remove the useless words.
I used some idea in the [kaggle forum](https://www.kaggle.com/ibtesama/getting-started-with-a-movie-recommendation-system), which is the cosine similarity between the documents.

So, I change my idea. `I would find the top related movies in the dataset and return the genre which appears the most`.
Please refer to the `predict` function in `model.py`.
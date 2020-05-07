import sys
import nltk
import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from sklearn.model_selection import GridSearchCV
import os
import pickle

nltk.download(["punkt", "wordnet", "stopwords"])


def load_data(database_filepath):
    """"

    Loads data needed for Disaster Response models

    params:
    database_filepath: database file to read cleaned data from


    returns:
    X: dataframe of values to fit on model
    Y: Message classifications dataframe
    categories:  List of possible categories classification

    """
    base = "sqlite:///"
    engine = create_engine(os.path.join(base, database_filepath))
    table = os.path.basename(database_filepath).split(".")[0]
    df = pd.read_sql("SELECT * FROM {}".format(table), engine)
    X = df["message"].values
    Y = df.iloc[:, 4:].values
    categories = df.iloc[:, 4:].columns.values
    return X, Y, categories


def tokenize(text):
    """"

    Preprocesses text input string

    params:
    text: text to preprocess

    returns:
    clean_tokens: list of text input preprocessed

    """
    stop_words = stopwords.words("english")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [
        lemmatizer.lemmatize(word).lower().strip()
        for word in tokens
        if word not in stop_words
    ]

    return clean_tokens


# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


def build_model():
    """"

    Builds ML Model to train dataframe on

    params:
    None

    returns:
    cv: Pipeline model based with GridSearch parameters for best model

    """
    pipeline = Pipeline(
        [
            ("vect", CountVectorizer(tokenizer=tokenize)),
            ("tfidf", TfidfTransformer()),
            (
                "clf",
                MultiOutputClassifier(
                    KNeighborsClassifier(n_neighbors=15, weights="uniform")
                ),
            ),
        ]
    )
    parameters = {
        "clf__estimator__n_neighbors": [3, 5, 11, 15, 20],
        "clf__estimator__weights": ["uniform", "distance"],
    }
    cv = GridSearchCV(estimator=pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """"

    Evaluates trained model accuracy on fitted X_test data

    params:
    model: Pipelined GridSearch model
    X_test: messages to predict category
    Y_test: ground value truths to test accuracy against
    categories_names: list of possible category names

    returns:
    None

    """
    y_pred = model.predict(X_test)
    i = 0
    for name in category_names:
        true = np.take(Y_test, i, axis=1)
        pred = np.take(y_pred, i, axis=1)
        print("---------")
        print(name)
        print(classification_report(true, pred))
        i += 1


def save_model(model, model_filepath):
    """"

    Saves model as pickle file

    params:
    model: Pipelined GridSearch model
    model_filepath: final outputted model filepath + filename

    returns:
    None

    """
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    """"

    Runs classification flow:
    Reads clean messages + categories dataframe from db file
    Tokenizes text data
    Creates Pipelined GridSearch model to fit data on

    Python script call
    params:
    database_filepath: database file to read messages from
    model_filepath: final output classifier destination desired

    Returns:
    None

    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print("Loading data...\n    DATABASE: {}".format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print("Building model...")
        model = build_model()

        print("Training model...")
        model.fit(X_train, Y_train)

        print("Evaluating model...")
        evaluate_model(model, X_test, Y_test, category_names)

        print("Saving model...\n    MODEL: {}".format(model_filepath))
        save_model(model, model_filepath)

        print("Trained model saved!")

    else:
        print(
            "Please provide the filepath of the disaster messages database "
            "as the first argument and the filepath of the pickle file to "
            "save the model to as the second argument. \n\nExample: python "
            "train_classifier.py ../data/DisasterResponse.db classifier.pkl"
        )


if __name__ == "__main__":
    main()

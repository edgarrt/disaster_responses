import json
import plotly
import pandas as pd
import nltk
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar, Scatter, Table
from sklearn.externals import joblib
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os

nltk.download(["punkt", "wordnet", "stopwords"])


app = Flask(__name__)

# No stopword removal
# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


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


# load data
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
db_file = os.path.join(project_root, "data", "DisasterResponse.db")
base = "sqlite:///"
file_to_open = base + db_file
engine = create_engine(file_to_open)
df = pd.read_sql_table("DisasterResponse", engine)

# load model
model_filepath = os.path.join(project_root, "models", "classifier.pkl")
model = joblib.load(model_filepath)


def get_prediction_results(message):
    """"

    Tokenizes inputted paramets and guess category classification

    params:
    mesage: message to predict

    returns:
    predicted: Array classfications predicted

    """
    result = model.predict([message])[0]
    result = dict(zip(df.columns[4:], result))
    predicted = []
    for category, classification in result.items():
        if classification == 1:
            predicted.append(category)
    return predicted


def render_graphs():
    """"

    Defines graphs to be rendered

    params:
    None

    returns:
    None

    """

    genre_counts = df.groupby("genre").count()["message"]
    genre_names = list(genre_counts.index)

    graph_one = []
    graph_one.append(Bar(x=genre_names, y=genre_counts))
    layout_one = {
        "title": "Distribution of Message Genres",
        "yaxis": {"title": "Count"},
        "xaxis": {"title": "Genre"},
    }

    #     # Total counts of seen categories
    df_counts = df.iloc[:, 4:]
    df_counts = df_counts.apply(pd.Series.value_counts)
    df_counts = df_counts.iloc[1]
    categories = list(df.iloc[:, 4:])

    graph_two = []
    graph_two.append(Bar(x=categories, y=df_counts))
    layout_two = {
        "title": "Distribution of Message Categories",
        "yaxis": {"title": "Count"},
        "xaxis": {"title": "Category"},
    }

    df_counts_percentage = df_counts.apply(lambda x: x / 26180)
    graph_three = []
    graph_three.append(
            Scatter(x=categories, y=df_counts_percentage, mode="markers")
        )
    layout_three = {
        "title": "Category Occurance Percentage in Dataset",
        "yaxis": {"title": "Occurance percentage in rows"},
        "xaxis": {"title": "Category"},
    }

    # sample predictions
    example_1 = "We are more than 50 people sleeping on the street. Please help us find tent, food."
    predicted_1 = ["related", "request", "aid_related", "shelter", "direct_report"]
    example_2 = "where is the shelter location?"
    predicted_2 = ["related", "aid_related", "shelter"]

    graph_four = []
    graph_four.append(
        Table(
            header=dict(values=["Input", "Classification"]),
            cells=dict(
                values=[
                [example_1, example_2], 
                [str(predicted_1), str(predicted_2)]
                ]
            ),
        )
    )
    layout_four = {"title": "Category Occurance Percentage in Dataset", "height": 275}

    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))

    return figures


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
    """"

    Routes intial endpoints

    params:
    None

    returns:
    None

    """
    # render custom graphs created
    graphs = render_graphs()
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template("master.html", ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route("/go")
def go():
    """"

    Routes query endpoint for user inputted message string

    params:
    query: message to guess

    returns:
    None

    """
    # save user input in query
    query = request.args.get("query", "")

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        "go.html", query=query, classification_result=classification_results
    )


def main():
    port = int(os.environ.get("PORT", 3001))
    app.run(host="0.0.0.0", port=port, debug=True)


if __name__ == "__main__":
    main()

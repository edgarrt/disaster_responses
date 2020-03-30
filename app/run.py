import json
import plotly
import pandas as pd
import nltk
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Scatter, Table
import plotly.graph_objs as go
from sklearn.externals import joblib
from sqlalchemy import create_engine
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download(["punkt", "wordnet", "stopwords"])


app = Flask(__name__)

# def tokenize(text):
#     tokens = word_tokenize(text)
#     lemmatizer = WordNetLemmatizer()

#     clean_tokens = []
#     for tok in tokens:
#         clean_tok = lemmatizer.lemmatize(tok).lower().strip()
#         clean_tokens.append(clean_tok)

#     return clean_tokens


def tokenize(text):
    stop_words = stopwords.words("english")

    #     text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = [
        lemmatizer.lemmatize(word).lower().strip()
        for word in tokens
        if word not in stop_words
    ]

    return clean_tokens


# load data
engine = create_engine("sqlite:///../data/DisasterResponse.db")
df = pd.read_sql_table("DisasterResponse", engine)

# load model
model = joblib.load("../models/classifier.pkl")


def get_prediction_results(message):
    result = model.predict([message])[0]
    result = dict(zip(df.columns[4:], result))
    predicted = []
    for category, classification in result.items():
        if classification == 1:
            predicted.append(category)
    return predicted


def render_graphs():
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals

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
    graph_three.append(Scatter(x=categories, y=df_counts_percentage, mode="markers"))
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
                values=[[example_1, example_2], [str(predicted_1), str(predicted_2)]]
            ),
        )
    )
    layout_four = {"title": "Category Occurance Percentage in Dataset", "height": 275}

    #     graph_five = []
    #     graph_five.append(
    #                   Table(header=dict(values=df.columns),
    #                         cells=dict(values=df.loc[0,:]
    #                                   )
    #                        )
    #     )
    #     layout_five =  {
    #         'title': 'Category Occurance Percentage in Dataset',
    #     }

    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    figures.append(dict(data=graph_four, layout=layout_four))
    #     figures.append(dict(data=graph_five, layout=layout_five))

    return figures


# index webpage displays cool visuals and receives user input text for model
@app.route("/")
@app.route("/index")
def index():
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
    app.run(host="0.0.0.0", port=3001, debug=True)


if __name__ == "__main__":
    main()
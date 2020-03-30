import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine
import os


def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=("id"))
    return df


def clean_data(df):
    categories = df["categories"].str.split(";", expand=True)
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: pd.Series(str(x).split("-"))[0])

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))

    del df["categories"]

    df = pd.concat([df, categories], axis=1)

    duplicated = df[df.duplicated(["id"])]
    print(len(duplicated))
    df = df.drop_duplicates(subset="id", keep="first")

    return df


def save_data(df, database_filename):
    print("Saving dataframe")
    base = "sqlite:///"
    engine = create_engine(os.path.join(base, database_filename))
    table = os.path.basename(database_filename).split(".")[0]
    df.to_sql(table, engine, index=False)
    print("saved to {} table".format(table))


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print(
            "Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}".format(
                messages_filepath, categories_filepath
            )
        )
        df = load_data(messages_filepath, categories_filepath)

        print("Cleaning data...")
        df = clean_data(df)

        print("Saving data...\n    DATABASE: {}".format(database_filepath))
        save_data(df, database_filepath)

        print("Cleaned data saved to database!")

    else:
        print(
            "Please provide the filepaths of the messages and categories "
            "datasets as the first and second argument respectively, as "
            "well as the filepath of the database to save the cleaned data "
            "to as the third argument. \n\nExample: python process_data.py "
            "disaster_messages.csv disaster_categories.csv "
            "DisasterResponse.db"
        )


if __name__ == "__main__":
    main()

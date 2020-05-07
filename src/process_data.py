import sys
import pandas as pd
import sqlite3
from sqlalchemy import create_engine
import os
import logging


def load_data(messages_filepath, categories_filepath):
    """"

    Loads data needed for Disaster Response models
    params:
    messages_filepath: path to messages csv file
    categories_filepath: path to categories csv file

    returns:
    df:  merged dataframes of messages + categories csvs

    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on=("id"))
    return df


def clean_data(df):
    """"
    Preprocesses the data to be split and then fitted to model

    params:
    df:  merged dataframes of messages + categories csvs

    returns:
    df: cleaned dataframe of messages + categories csvs with duplicates removed

    """

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
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """"
    Saves dataframe passed to correspding db file passed

    params:
    df:  dataframe to save
    database_filename: filepath of database file to save to

    returns:
    None

    """

    base = "sqlite:///"
    try:
        file_to_open = base + database_filename
        engine = create_engine(file_to_open)
    except sqlite3.OperationalError:
        print("Error saving processed data")
        sys.exit(1)
    table = os.path.basename(database_filename).split(".")[0]
    engine.execute("""DROP TABLE {}""".format(table))
    df.to_sql(table, engine, index=False)
    print("saved to {} table".format(table))


def main():
    """"

    Runs preprocessing flow:
    Reads messages + categories csvs
    Cleans data
    Then saves to database file

    Python script call
    params:
    messages_filepath: messages csv
    categories_filepath: categories csv
    database_filepath: output database file desired

    Returns:
    None

    """

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

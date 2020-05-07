# Disaster Response Pipeline Project
### Project Summary
- Created web app that utilizes machine learning model trained on data set containing real messages that were sent during disaster events.
- Created a machine learning pipeline that automatically categorizes messages received


### Project components
#### 1. ETL Pipeline
- data cleaning pipeline Python script, process_data.py:
  - Loads the messages and categories datasets
  - Merges the two datasets
  - Cleans the data
  - Stores it in a SQLite database

#### 2. ML Pipeline
- machine learning pipeline Python script, train_classifier.py:
  - Loads data from the SQLite database
  - Splits the dataset into training and test sets
  - Builds a text processing and machine learning pipeline
  - Trains and tunes a model using GridSearchCV
  - Outputs results on the test set
  - Exports the final model as a pickle file

#### 3. Flask Web App
- flask, html, css and javascript


#### 4. Dockerfile
- Builds image and pushes to heroku hosting

### View it live
https://gentle-retreat-29205.herokuapp.com/

### Instructions to run locally:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python src/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python src/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python app/run.py`

3. Go to http://0.0.0.0:3001/

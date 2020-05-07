FROM python:3.6.8

COPY requirements.txt ./
RUN pip install --force --ignore-installed -r requirements.txt

ENV PORT=80
EXPOSE 80

# TODO: uncomment when fixed train_classifier.py
CMD mkdir -p models

COPY app/ app/
COPY data/ data/
COPY src/ src/


#RUN python src/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db

# TODO: Investigate following call hang, workaround copying models to container
#RUN python src/train_classifier.py data/DisasterResponse.db models/classifier.pkl

COPY models/ models/
CMD python app/run.py

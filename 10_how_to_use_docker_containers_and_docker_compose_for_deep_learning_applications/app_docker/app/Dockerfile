FROM tensorflow/tensorflow:2.9.3

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

CMD ["uwsgi", "app.ini"]

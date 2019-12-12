FROM python:3.7.3-slim-stretch

RUN apt-get -y update && apt-get -y install gcc

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME

#COPY . ./

COPY models/text_model/* /app/models/text_model/

COPY app.py /app
COPY encoder.py /app
COPY generate.py /app
COPY model.py /app
COPY sample.py /app

# Make changes to the requirements/app here.
# This Dockerfile order allows Docker to cache the checkpoint layer
# and improve build times if making changes.
RUN pip3 --no-cache-dir install tensorflow==1.14.0 gunicorn flask regex

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

#ENTRYPOINT ["python3", "-X", "utf8", "app.py"]

CMD exec gunicorn --bind :$PORT --workers 1 --threads 1 app

FROM python:3.7.3-slim-stretch

RUN apt-get -y update && apt-get -y install gcc

WORKDIR /
COPY models/text_model/ /models/text_model/

# Make changes to the requirements/app here.
# This Dockerfile order allows Docker to cache the checkpoint layer
# and improve build times if making changes.
RUN pip3 --no-cache-dir install tensorflow==1.14.0 starlette uvicorn flask Flask request jsonify regex

COPY app.py app.py
COPY encoder.py encoder.py
COPY generate.py generate.py
COPY model.py model.py
COPY sample.py sample.py

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["python3", "-X", "utf8", "app.py"]

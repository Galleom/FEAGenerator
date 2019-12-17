FROM python:3.7.3-slim-stretch

RUN apt-get -y update && apt-get -y install gcc

COPY models/text_model/* /app/models/text_model/

# Make changes to the requirements/app here.
# This Dockerfile order allows Docker to cache the checkpoint layer
# and improve build times if making changes.
RUN pip3 --no-cache-dir install tensorflow==1.15.0 gunicorn starlette uvicorn ujson requests regex

COPY app.py /
COPY encoder.py /
COPY generate.py /
COPY model.py /
COPY sample.py /

# Clean up APT when done.
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

ENTRYPOINT ["python3", "-X", "utf8", "app.py"]

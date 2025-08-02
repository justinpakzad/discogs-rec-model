FROM python:3.11.0-slim

WORKDIR /discogs_rec

RUN apt-get update && apt-get install -y \
    build-essential \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /discogs_rec/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY discogs_rec/main.py /discogs_rec/main.py
COPY discogs_rec/preprocessing.py /discogs_rec/preprocessing.py
COPY discogs_rec/utils.py /discogs_rec/utils.py


CMD ["python" ,"main.py"]
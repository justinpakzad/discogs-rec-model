FROM python:3.12.0-slim
WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y \
    build-essential \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the package structure
COPY discogs_rec/__init__.py ./discogs_rec/__init__.py
COPY discogs_rec/main.py ./discogs_rec/main.py
COPY discogs_rec/preprocessing.py ./discogs_rec/preprocessing.py
COPY discogs_rec/utils.py ./discogs_rec/utils.py
COPY tests ./tests

CMD ["python", "-m", "discogs_rec.main"]
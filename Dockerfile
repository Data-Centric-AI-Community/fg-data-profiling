FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "setuptools>=72.0.0,<80.0.0" wheel && \
    pip install --no-cache-dir . && \
    pip install --no-cache-dir "setuptools>=72.0.0,<80.0.0" && \
    pip install --no-cache-dir jupyter

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]



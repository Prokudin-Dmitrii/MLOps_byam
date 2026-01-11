FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    ca-certificates \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY model/ model/
COPY data/tokenizer/ data/tokenizer/
COPY config_params.yaml .

ENTRYPOINT ["python", "-m", "src.predict"]
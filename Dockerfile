FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ libxml2-dev libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONPATH=/app

COPY requirements.txt /app/requirements.txt
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY . /app
RUN mkdir -p /app/.storage /app/logs

ENV PROMPTS_DIR=/app/prompts \
    DB_PATH=/app/.storage/research.db \
    CHROMA_PATH=/app/.storage/.chroma \
    LOG_DIR=/app/logs

CMD ["python", "agent.py", "--prompt", "What are the main LLM alignment techniques developed in 2024?"]

# Smart Research Agent

## How to Run Without Docker

```bash
# 1. Create virtual environment and activate it
python3 -m venv .venv && source .venv/bin/activate

# 2. Upgrade pip
pip install --upgrade pip

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env  # fill in keys
export $(grep -v '^#' .env | xargs)

# 5. Run the agent
python agent.py --prompt "What are the main LLM alignment techniques developed in 2024?"
```

---

## Running Tests (81% Coverage)

```bash
python -m pytest --cov=agent --cov-report=term-missing
```

---

## 🚀 Docker (Compose) Quick Start

### 1. Prerequisites

- **Docker Desktop** installed and running  
- **Python is NOT required** on your host — everything runs inside Docker  
- `.env` file with your API keys and config

```bash
cp .env.example .env
# Edit .env and set your keys
```

---

### 2. Build the Docker Image

```bash
docker compose build
```

---

### 3. Run the Agent

#### Interactive CLI (type queries until exit)
```bash
docker compose run --rm agent
```

#### One-Shot Query (non-interactive)
```bash
docker compose run --rm agent-once \
  python agent.py --prompt "Give me an executive-level brief on the newest LLM alignment techniques."
```

#### With Parameters
```bash
docker compose run --rm agent-once \
  python agent.py \
  --prompt "Executive brief on vision-language models 2024–2025" \
  --max-fetch 12 \
  --k-retrieve 8 \
  --min-local-sources 3
```

**Parameter explanations:**
- `--prompt` — Your research question or query (string)  
- `--max-fetch` — Maximum number of sources to fetch from the web/ArXiv before synthesis  
- `--k-retrieve` — Number of top-ranked chunks to retrieve for LLM synthesis  
- `--min-local-sources` — Minimum number of locally cached sources required before retrieval (will fetch if fewer exist)

---

### 4. Run Tests & Coverage in Docker

```bash
docker compose run --rm tests
```

---

### 5. Rebuild After Code Changes

```bash
docker compose build --no-cache
```

---

## LangSmith Observability

Project link *(private)*:  
[LangSmith Project Dashboard](https://smith.langchain.com/o/4b3bfd83-78a2-47d9-8a39-03be0da9d878/projects/p/d69655cf-1845-461e-bec7-fa717b713a2d?timeModel=%7B%22duration%22%3A%227d%22%7D)  
> I think LangSmith projects can only be shared via **email invite**. Check [official guide](https://docs.smith.langchain.com/observability/how_to_guides/share_trace).

So, I share some traces examples **public traces**:
1. [Trace Example 1](https://smith.langchain.com/public/48028d15-8d29-4ae0-be28-13adca350810/r)
2. [Trace Example 2](https://smith.langchain.com/public/a18b2289-787b-499a-85bc-5426166d474c/r)

---

## Demo Video

[Watch the Demo](https://drive.google.com/file/d/1O5msWegq_I5EDYUxn9OweUbe8W8V6D9V/view?usp=sharing)

---


Link to Langsmith:  https://smith.langchain.com/o/4b3bfd83-78a2-47d9-8a39-03be0da9d878/projects/p/d69655cf-1845-461e-bec7-fa717b713a2d?timeModel=%7B%22duration%22%3A%227d%22%7D 

I do not think it's possible to share the whole project link, only via email invite: https://docs.smith.langchain.com/observability/how_to_guides/share_trace

But here are examples of two traces: 
    1. https://smith.langchain.com/public/48028d15-8d29-4ae0-be28-13adca350810/r
    2. https://smith.langchain.com/public/a18b2289-787b-499a-85bc-5426166d474c/r

How to run with Docker:
TODO


Video: https://drive.google.com/file/d/1O5msWegq_I5EDYUxn9OweUbe8W8V6D9V/view?usp=sharing 






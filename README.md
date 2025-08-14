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

# 6. Running Tests (81% Coverage)
python -m pytest --cov=agent --cov-report=term-missing
```

---

## Docker (Compose) Quick Start

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
## Agent Architecture & Design Choices

The **Smart Research Agent** is a headless CLI-based research assistant designed for efficient, repeatable research workflows. This agent was designed for **fast, repeatable, and grounded research**, balancing **accuracy**, **cost**, and **latency**.

**Core architecture:**

- **CLI Entry Point (`agent.py`)** – Handles argument parsing, interactive mode, and one-shot queries.
- **Logging & Observability** – Structured JSON logs written via `RotatingFileHandler` and integrated with **LangSmith** for tracing runs, spans, and LLM calls.
- **Storage Layer**  
  - **SQLite** – Stores source metadata, canonicalized URLs, deduplication hashes, and cleaned text.  
  - **ChromaDB** – Persistent vector store for semantic retrieval of text chunks.
- **Planning** – Uses an LLM (`planner_system.txt` + `planner_user.txt`) to generate targeted search queries for both **web** and **arXiv** sources.
- **Fetching & Parsing** –  
  - Web search via **SERPAPI**  
  - Academic search via **arXiv API**  
  - Fetches raw HTML/PDF, extracts clean text with **Trafilatura** + **BeautifulSoup**, or **PyPDF** for PDFs.
- **Indexing** – Chunks text (~900 tokens), generates embeddings via OpenAI, deduplicates near-identical chunks, and indexes them in ChromaDB.
- **Memory Management** – Before running searches, checks local DB for semantically similar sources to avoid unnecessary network calls.
- **Prompting for Synthesis** –  
  - **Planner Prompts** – Guide LLM to propose relevant search queries.  
  - **Synthesis Prompts** – Combine retrieved evidence chunks with metadata to produce a well-cited, structured markdown answer.
- **Final Output** – Markdown with:
  1. Structured answer
  2. Inline citations (ordinal numbers)
  3. Reference list with URLs, authors, and publication years.


**Design Rationale**

**Planning - LLM-driven query generation**  
- **Why:** A static query often misses context or nuances of the user’s request.  
- **Choice:** Use `planner_system.txt` + `planner_user.txt` prompts to let an LLM generate focused web/arXiv search queries.  
- **Trade-off:** Adds one LLM call per run, but improves relevance of sources.

**Memory - local semantic cache**  
- **Why:** Avoid fetching the same information repeatedly.  
- **Choice:** Store processed sources in **SQLite** (metadata/text) and **ChromaDB** (vector embeddings) for semantic matching.  
- **Trade-off:** Slight storage overhead, but large latency/cost savings for repeated or related queries.

**Prompting (structured system + user templates)**  
- **Why:** Consistent answer structure and reliable citation grounding.  
- **Choice:**  
  - **Planner prompts** to shape search strategy.  
  - **Synthesis prompts** to instruct LLM to weave retrieved evidence into markdown with inline citations and a reference list.  
- **Trade-off:** Requires maintaining separate prompt files, but makes prompt iteration and tuning easier.

### **Query -> Answer Flow (High-level)**

```text
[1] User prompt (CLI)
        ↓
[2] Local memory check (ChromaDB semantic match)
    ├── Enough local matches → Skip search
    └── Not enough → LLM plans search queries
            ↓
[3] Search web (SERPAPI) + arXiv
        ↓
[4] Merge + deduplicate results
        ↓
[5] Fetch + parse (Trafilatura / PyPDF)
        ↓
[6] Store in SQLite + index in ChromaDB
        ↓
[7] Retrieve top-k relevant chunks (ChromaDB)
        ↓
[8] Build context block with snippets + citations
        ↓
[9] Synthesis prompt to LLM
        ↓
[10] Output: Structured markdown + References
```
**Key benefits of the system flow:**  
- **Early exit on local coverage**: saves API calls and speeds up repeat queries.  
- **LLM planning step**: better search recall and precision.  
- **Evidence-first synthesis**: keeps answers grounded and citeable.

---

## Trade-offs & What I’d Tackle Next

**Trade-offs in current implementation:**

- **LLM-driven planning**  
  - Pros: Adaptable to diverse prompts, dynamically generates relevant search queries.  
  - Cons: Dependent on LLM accuracy; occasional off-topic queries.
- **Single vector store (ChromaDB)**  
  - Pros: Simple, persistent, fast retrieval for prototyping.  
  - Cons: No sharding or distributed scaling; less optimal for very large datasets.
- **OpenAI embeddings**  
  - Pros: High quality, plug-and-play.  
  - Cons: Adds latency and API cost; no offline embedding generation.

**Next steps and improvements:**

1. **Improve planning reliability** – Add fallback query templates when LLM output is incomplete or off-topic.
2. **Add async fetching** – Parallelize web and PDF fetches to reduce latency.
3. **Hybrid retrieval** – Combine keyword-based search with semantic retrieval for more diverse evidence.
4. **Multi-model synthesis** – Experiment with other LLMs (Claude, Mistral) for synthesis quality and cost trade-offs.
5. **Web UI mode** – Wrap CLI agent in a lightweight Flask/FastAPI server for easier non-technical use.

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
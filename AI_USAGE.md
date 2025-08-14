# AI_USAGE.md

This file is just a running log of stuff I adapted from AI tools, docs, YouTube, or repos while building this project.

---

## ChatGPT (GPT-5) snippets I used

- Gave me some starter code patterns based on my architecture so I could build out core functions faster.
- Helped me write the CLI argument parsing with `argparse` pasted into `agent.py`.
- Gave me example code for structured JSON logging with `logging` + `RotatingFileHandler` adapted into `observability.py`.
- Wrote test scaffolding with `pytest` and `pytest-cov` put into `tests/test_cli.py` and `tests/test_agent.py`.
- URL cleaning snippet with `urllib.parse` to strip UTM/tracking params in `utils.py` as `clean_url()`.
- README sections for setup, running agent, running unit tests, and Docker copied into `README.md`.
- Gave me code for parallel fetching of sources and retry logic using `tenacity`.
- Helped with Docker setup basics and adding run instructions to the README.
- Helped me understand and implement logging of LLM calls and LangSmith tracing integration.
- Helped to formulate my Smart Research Agent architecture into a clearer step-by-step plan, which I put in `README.md`.

---

## Other sources I used

### YouTube videos
- https://www.youtube.com/watch?v=EgpLj86ZHFQ&list=WL  
- https://www.youtube.com/watch?v=3c-iBn73dDE&t=1554s  
- https://www.youtube.com/watch?v=tFXm5ijih98&t=234s  
- https://www.youtube.com/watch?v=jx7xuHlfsEQ&t=72s  
- https://www.youtube.com/watch?v=e9P56k-x8B4  

### GitHub repo
- https://github.com/pixegami/rag-tutorial-v2/tree/main

---

How to run without Docker:

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env  # fill in keys
export $(grep -v '^#' .env | xargs)

python agent.py --prompt "What are the main LLM alignment techniques developed in 2024?"

To run Tests (81% coverage):
python -m pytest --cov=agent --cov-report=term-missing 


Link to Langsmith:  https://smith.langchain.com/o/4b3bfd83-78a2-47d9-8a39-03be0da9d878/projects/p/d69655cf-1845-461e-bec7-fa717b713a2d?timeModel=%7B%22duration%22%3A%227d%22%7D 

I do not think it's possible to share the whole project link, only via email invite: https://docs.smith.langchain.com/observability/how_to_guides/share_trace

But here are examples of two traces: 
    1. https://smith.langchain.com/public/48028d15-8d29-4ae0-be28-13adca350810/r
    2. https://smith.langchain.com/public/a18b2289-787b-499a-85bc-5426166d474c/r

How to run with Docker:
TODO


Video: https://drive.google.com/file/d/1O5msWegq_I5EDYUxn9OweUbe8W8V6D9V/view?usp=sharing 






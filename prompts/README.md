# Prompt Packet

This folder contains all prompt templates used by the Smart Research Agent.

## Files

- planner_system.txt
  System role for the "planning" LLM call. Constrains output to JSON only.

- planner_user.txt
  User message template for the "planning" call.
  Variables:
    - {{prompt}} : The user's research question.

- synthesis_system.txt
  System role for the "synthesis" LLM call.

- synthesis_user.txt
  User message template for the "synthesis" call.
  Variables:
    - {{prompt}}        : The user's research question.
    - {{context_block}} : Numbered snippets the model can cite.
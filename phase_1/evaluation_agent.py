"""Test script for the EvaluationAgent class.

Flow demonstrated here:
  - A KnowledgeAugmentedPromptAgent is set up with deliberately wrong knowledge
    and a verbose persona ("Dear students, ...").
  - An EvaluationAgent wraps it, enforcing the criterion that the answer must
    be the name of a city only (not a full sentence).
  - Because the worker's initial answer is a full sentence, the evaluator will
    loop: request corrections, feed them back, and re-evaluate until the worker
    produces a single city name or the interaction budget is exhausted.
"""

from workflow_agents.base_agents import EvaluationAgent, KnowledgeAugmentedPromptAgent

# Load the OpenAI (Vocareum) API key from the environment.
# Copy .env.example to .env at the project root and fill in OPENAPI_KEY
# before running this script.
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAPI_KEY")
if not openai_api_key:
    raise RuntimeError(
        "OPENAPI_KEY is not set. Copy .env.example to .env and fill it in, "
        "or `export OPENAPI_KEY=voc-...` in your shell."
    )

prompt = "What is the capital of France?"

# Worker: persona + (intentionally incorrect) knowledge.
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

# Evaluator: enforces "name of a city only" as the output shape.
eval_persona = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=eval_persona,
    evaluation_criteria=evaluation_criteria,
    agent_to_evaluate=knowledge_agent,
    max_interactions=10,
)

result = evaluation_agent.evaluate(prompt)

print("\n=== Final evaluation result ===")
print(result)

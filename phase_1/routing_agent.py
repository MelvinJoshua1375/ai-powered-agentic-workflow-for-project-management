"""Test script for the RoutingAgent class.

Three KnowledgeAugmentedPromptAgents (Texas, Europe, Math) are wrapped as routes.
The RoutingAgent embeds each incoming prompt and each route description with
``text-embedding-3-large`` and dispatches to the highest cosine-similarity match.
"""

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

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

# Three specialist knowledge agents ---------------------------------------
persona = "You are a college professor"

texas_knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, texas_knowledge)

europe_knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, europe_knowledge)

math_persona = "You are a college math professor"
math_knowledge = (
    "You know everything about math, you take prompts with numbers, "
    "extract math formulas, and show the answer without explanation"
)
math_agent = KnowledgeAugmentedPromptAgent(openai_api_key, math_persona, math_knowledge)

# Routing agent ------------------------------------------------------------
routing_agent = RoutingAgent(openai_api_key, [])
routing_agent.agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x),
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x),
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.respond(x),
    },
]

# Route and print the response for each of the three sample prompts.
prompts = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories",
]

for test_prompt in prompts:
    print(f"\n>>> Prompt: {test_prompt}")
    print(routing_agent.route(test_prompt))

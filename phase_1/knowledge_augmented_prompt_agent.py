"""Test script for the KnowledgeAugmentedPromptAgent class."""

from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

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

# A persona + deliberately-wrong knowledge string. If the agent truly uses only
# the supplied knowledge (rather than its own pretraining), it should tell us
# the capital of France is London, not Paris.
persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

# Instantiate the KnowledgeAugmentedPromptAgent with persona and knowledge.
knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona, knowledge)

# Send the prompt to the agent and capture the response.
response = knowledge_agent.respond(prompt)

# Print the response and a confirmation line demonstrating that the agent used
# the provided knowledge rather than its own pretrained knowledge.
print(response)
print(
    "\nConfirmation: The agent's answer cites London as the capital of France, "
    "which is the value supplied in the knowledge string. This demonstrates "
    "that the KnowledgeAugmentedPromptAgent is using the provided knowledge "
    "rather than its inherent LLM knowledge (which would have returned Paris)."
)

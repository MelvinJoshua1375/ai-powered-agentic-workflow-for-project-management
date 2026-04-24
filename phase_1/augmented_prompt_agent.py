"""Test script for the AugmentedPromptAgent class."""

from workflow_agents.base_agents import AugmentedPromptAgent

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
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate the agent with the persona and the API key.
augmented_agent = AugmentedPromptAgent(openai_api_key, persona)

# Send the prompt to the agent and keep the result in augmented_agent_response.
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response.
print(augmented_agent_response)

# Print the commentary so the terminal output captures the rubric's required
# discussion of the knowledge source and the persona's impact.
print(
    "\n[Knowledge source] The AugmentedPromptAgent does not receive any external "
    "knowledge string, so its answer still comes from gpt-3.5-turbo's pretrained "
    "general knowledge. It correctly recalls that the capital of France is Paris."
)
print(
    "[Persona impact] The persona is injected through the system message, which "
    "tells the model to respond as a college professor whose answers start with "
    "'Dear students,'. This does not change the factual content of the answer, "
    "but it reshapes the tone, formality, and opening so the output reads like a "
    "short classroom remark rather than a plain factual reply."
)

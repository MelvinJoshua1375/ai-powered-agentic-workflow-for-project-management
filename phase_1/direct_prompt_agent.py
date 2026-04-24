"""Test script for the DirectPromptAgent class."""

from workflow_agents.base_agents import DirectPromptAgent

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

prompt = "What is the Capital of France?"

# Instantiate the agent and call its primary method with the sample prompt.
direct_agent = DirectPromptAgent(openai_api_key)
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent.
print(direct_agent_response)

# Explain the knowledge source used by the DirectPromptAgent.
# The DirectPromptAgent sends the prompt to gpt-3.5-turbo with no system message,
# no persona, and no supplementary knowledge. The answer therefore comes purely
# from the LLM's own pretraining data (its general, world knowledge learned
# during training). There is no retrieval step and no user-supplied context.
print(
    "\n[Knowledge source] The agent answered using the general world knowledge "
    "encoded in the gpt-3.5-turbo model during pretraining. No external context, "
    "retrieved documents, or custom persona were supplied to the agent."
)

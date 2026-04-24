"""agentic_workflow.py

General-purpose AI-powered agentic workflow for technical project management
at InnovateNext Solutions. Pilot input: the Email Router product specification.

Pipeline
--------
1. ``ActionPlanningAgent`` decomposes the TPM's high-level prompt into
   sub-tasks, grounded on its ``knowledge_action_planning`` block.
2. ``RoutingAgent`` assigns each sub-task to one of three "team" support
   functions (Product Manager / Program Manager / Development Engineer) using
   semantic similarity between the sub-task and each team's description.
3. Each team = ``KnowledgeAugmentedPromptAgent`` (worker) + ``EvaluationAgent``
   (judge-and-refiner). The support function runs the worker, passes its first
   answer to the judge, and returns the final validated response.
4. The completed step results are collected; the final workflow output is the
   last completed step (typically the engineering task list).
"""

import os
from dotenv import load_dotenv

# 1. Import the four agent classes used by the workflow.
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)

# 2. Load the OpenAI (Vocareum) API key from the environment.
# Copy .env.example to .env at the project root and fill in OPENAI_API_KEY
# before running this workflow.
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in, "
        "or `export OPENAI_API_KEY=voc-...` in your shell."
    )

# 3. Load the Email Router product specification. Resolved relative to this
# script so the workflow can be run from any working directory.
_spec_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "Product-Spec-Email-Router.txt"
)
with open(_spec_path, "r", encoding="utf-8") as _f:
    product_spec = _f.read()


# ---------------------------------------------------------------------------
# 4. Action Planning Agent
# ---------------------------------------------------------------------------
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)


# ---------------------------------------------------------------------------
# 5-7. Product Manager team (knowledge worker + evaluator)
# ---------------------------------------------------------------------------
persona_product_manager = (
    "You are a Product Manager, you are responsible for defining the user stories for a product."
)
# TODO 5: append the loaded product_spec so the worker has the spec at hand.
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product.\n\n"
    "Product Specification:\n"
    f"{product_spec}"
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_product_manager, knowledge_product_manager
)

persona_product_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents"
)
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=(
        "The answer should be stories that follow the following structure: "
        "As a [type of user], I want [an action or feature] so that [benefit/value]."
    ),
    agent_to_evaluate=product_manager_knowledge_agent,
    max_interactions=5,
)


# ---------------------------------------------------------------------------
# 8. Program Manager team
# ---------------------------------------------------------------------------
persona_program_manager = (
    "You are a Program Manager, you are responsible for defining the features for a product."
)
knowledge_program_manager = (
    "Features of a product are defined by organizing similar user stories into cohesive groups."
)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_program_manager, knowledge_program_manager
)

persona_program_manager_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=(
        "The answer should be product features that follow the following structure: "
        "Feature Name: A clear, concise title that identifies the capability\n"
        "Description: A brief explanation of what the feature does and its purpose\n"
        "Key Functionality: The specific capabilities or actions the feature provides\n"
        "User Benefit: How this feature creates value for the user"
    ),
    agent_to_evaluate=program_manager_knowledge_agent,
    max_interactions=5,
)


# ---------------------------------------------------------------------------
# 9. Development Engineer team
# ---------------------------------------------------------------------------
persona_dev_engineer = (
    "You are a Development Engineer, you are responsible for defining the development tasks for a product."
)
knowledge_dev_engineer = (
    "Development tasks are defined by identifying what needs to be built to implement each user story."
)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key, persona_dev_engineer, knowledge_dev_engineer
)

persona_dev_engineer_eval = (
    "You are an evaluation agent that checks the answers of other worker agents."
)
development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=(
        "The answer should be tasks following this exact structure: "
        "Task ID: A unique identifier for tracking purposes\n"
        "Task Title: Brief description of the specific development work\n"
        "Related User Story: Reference to the parent user story\n"
        "Description: Detailed explanation of the technical work required\n"
        "Acceptance Criteria: Specific requirements that must be met for completion\n"
        "Estimated Effort: Time or complexity estimation\n"
        "Dependencies: Any tasks that must be completed first"
    ),
    agent_to_evaluate=development_engineer_knowledge_agent,
    max_interactions=5,
)


# ---------------------------------------------------------------------------
# 11. Support functions for each role
# ---------------------------------------------------------------------------
# A support function: (1) gets an initial response from its KnowledgeAgent,
# (2) passes that response into its EvaluationAgent, which iterates until the
# evaluation criteria are met (or the iteration budget is exhausted), and
# (3) returns the ``final_response`` from the evaluator's result dict.

def product_manager_support_function(query):
    """Run the Product Manager (user stories) pipeline for a workflow step."""
    response_from_knowledge_agent = product_manager_knowledge_agent.respond(query)
    evaluation_result = product_manager_evaluation_agent.evaluate(
        response_from_knowledge_agent
    )
    return evaluation_result["final_response"]


def program_manager_support_function(query):
    """Run the Program Manager (features) pipeline for a workflow step."""
    response_from_knowledge_agent = program_manager_knowledge_agent.respond(query)
    evaluation_result = program_manager_evaluation_agent.evaluate(
        response_from_knowledge_agent
    )
    return evaluation_result["final_response"]


def development_engineer_support_function(query):
    """Run the Development Engineer (tasks) pipeline for a workflow step."""
    response_from_knowledge_agent = development_engineer_knowledge_agent.respond(query)
    evaluation_result = development_engineer_evaluation_agent.evaluate(
        response_from_knowledge_agent
    )
    return evaluation_result["final_response"]


# ---------------------------------------------------------------------------
# 10. Routing Agent
# ---------------------------------------------------------------------------
routing_agent = RoutingAgent(openai_api_key, [])
routing_agent.agents = [
    {
        "name": "Product Manager",
        "description": (
            "Responsible for defining product personas and user stories only. "
            "Does not define features or tasks. Does not group stories."
        ),
        "func": lambda x: product_manager_support_function(x),
    },
    {
        "name": "Program Manager",
        "description": (
            "Responsible for defining product features by grouping related user stories. "
            "Does not define user stories or engineering tasks."
        ),
        "func": lambda x: program_manager_support_function(x),
    },
    {
        "name": "Development Engineer",
        "description": (
            "Responsible for defining detailed engineering tasks (IDs, titles, "
            "descriptions, acceptance criteria, estimates, dependencies) required "
            "to implement user stories. Does not define user stories or product features."
        ),
        "func": lambda x: development_engineer_support_function(x),
    },
]


# ---------------------------------------------------------------------------
# 12. Run the workflow
# ---------------------------------------------------------------------------
print("\n*** Workflow execution started ***\n")

workflow_prompt = "What would the development tasks for this product be?"
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"Extracted {len(workflow_steps)} step(s):")
for i, s in enumerate(workflow_steps, 1):
    print(f"  {i}. {s}")

completed_steps = []

for i, step in enumerate(workflow_steps, 1):
    print(f"\n=== Executing step {i}/{len(workflow_steps)} ===")
    print(f"Step: {step}")
    try:
        step_result = routing_agent.route(step)
    except Exception as exc:
        # Basic error handling: record the failure and continue with the rest
        # of the workflow rather than halting the entire run.
        step_result = f"[Step {i} failed: {exc}]"
        print(step_result)
    completed_steps.append(step_result)
    print(f"Result for step {i}:\n{step_result}")

print("\n\n*** Workflow execution complete ***\n")
print("Final output of the workflow:")
if completed_steps:
    print(completed_steps[-1])
else:
    print("(no steps were produced by the action planning agent)")

"""
workflow_agents.base_agents

Reusable agent classes for InnovateNext Solutions' AI-powered agentic workflow
for project management.

Each class encapsulates a different interaction pattern with a Large Language
Model (OpenAI ``gpt-3.5-turbo`` for chat and ``text-embedding-3-large`` for
embeddings) served through the Vocareum endpoint used by the course workspace.
"""

from openai import OpenAI
import numpy as np
import pandas as pd
import re
import csv
import uuid
from datetime import datetime


# Vocareum-hosted OpenAI proxy used by the course workspace.
# The key (passed in at instantiation) must be a Vocareum-issued ``voc-...`` key.
VOCAREUM_BASE_URL = "https://openai.vocareum.com/v1"


# ---------------------------------------------------------------------------
# DirectPromptAgent
# ---------------------------------------------------------------------------
class DirectPromptAgent:
    """Sends the user prompt straight to the LLM with no additional context.

    This is the simplest possible agent: no persona, no extra knowledge, no
    memory. It answers entirely from the LLM's pretrained general knowledge.
    """

    def __init__(self, openai_api_key):
        # Store the API key so every ``respond`` call can instantiate a client.
        self.openai_api_key = openai_api_key

    def respond(self, prompt):
        """Return the assistant text for ``prompt`` using gpt-3.5-turbo."""
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# AugmentedPromptAgent
# ---------------------------------------------------------------------------
class AugmentedPromptAgent:
    """Responds from the point of view of a specific persona.

    The persona is injected via the system message and the agent is explicitly
    told to forget any previous conversational context.
    """

    def __init__(self, openai_api_key, persona):
        self.openai_api_key = openai_api_key
        self.persona = persona

    def respond(self, input_text):
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {self.persona}. "
                        "Forget all previous context and respond only using this persona."
                    ),
                },
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# KnowledgeAugmentedPromptAgent
# ---------------------------------------------------------------------------
class KnowledgeAugmentedPromptAgent:
    """Persona + explicit, constrained knowledge.

    The agent is ordered to answer *only* from the provided ``knowledge`` string
    and to ignore its own pretrained knowledge. This is the building block used
    for the Product Manager / Program Manager / Development Engineer roles in
    the Phase 2 workflow.
    """

    def __init__(self, openai_api_key, persona, knowledge):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge

    def respond(self, input_text):
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context.\n"
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge}\n"
            f"Answer the prompt based on this knowledge, not your own."
        )
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# RAGKnowledgePromptAgent (provided)
# ---------------------------------------------------------------------------
class RAGKnowledgePromptAgent:
    """
    Retrieval-augmented generation agent.

    Chunks a body of text, embeds the chunks with ``text-embedding-3-large``,
    and answers questions using the most similar chunk as grounding context.
    """

    def __init__(self, openai_api_key, persona, chunk_size=2000, chunk_overlap=100):
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.unique_filename = (
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"
        )

    def get_embedding(self, text):
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one, vector_two):
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text):
        separator = "\n"
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "text": text[start:end],
                    "chunk_size": end - start,
                    "start_char": start,
                    "end_char": end,
                }
            )

            start = end - self.chunk_overlap
            chunk_id += 1

        with open(
            f"chunks-{self.unique_filename}", "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self):
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["text"].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding="utf-8", index=False)
        return df

    def find_prompt_in_knowledge(self, prompt):
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding="utf-8")
        df["embeddings"] = df["embeddings"].apply(lambda x: np.array(eval(x)))
        df["similarity"] = df["embeddings"].apply(
            lambda emb: self.calculate_similarity(prompt_embedding, emb)
        )

        best_chunk = df.loc[df["similarity"].idxmax(), "text"]

        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context.",
                },
                {
                    "role": "user",
                    "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}",
                },
            ],
            temperature=0,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# EvaluationAgent
# ---------------------------------------------------------------------------
class EvaluationAgent:
    """Iteratively judges a worker agent's output against evaluation criteria.

    On each iteration:

    1. The worker agent answers the current prompt.
    2. The evaluator LLM checks that answer against ``evaluation_criteria`` and
       replies starting with ``Yes`` or ``No`` plus a reason.
    3. If the answer passes, the loop exits early. Otherwise the evaluator LLM
       produces concrete correction instructions, which are folded into a
       refined prompt that is handed back to the worker.

    The loop is capped at ``max_interactions`` iterations.

    ``evaluate`` returns a dictionary with the final worker response, the final
    evaluation text, and the number of iterations actually performed.
    """

    def __init__(
        self,
        openai_api_key,
        persona,
        evaluation_criteria,
        agent_to_evaluate,
        max_interactions,
    ):
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.agent_to_evaluate = agent_to_evaluate
        self.max_interactions = max_interactions

    def evaluate(self, initial_prompt):
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        prompt_to_evaluate = initial_prompt
        response_from_worker = ""
        evaluation = ""
        iterations = 0

        for i in range(self.max_interactions):
            iterations = i + 1
            print(f"\n--- Interaction {iterations} ---")

            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.agent_to_evaluate.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                f"Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are {self.persona}."},
                    {"role": "user", "content": eval_prompt},
                ],
                temperature=0,
            )
            evaluation = response.choices[0].message.content.strip()
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("Final solution accepted.")
                break

            print(" Step 4: Generate instructions to correct the response")
            instruction_prompt = (
                f"Provide instructions to fix an answer based on these reasons "
                f"why it is incorrect: {evaluation}"
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are {self.persona}. Produce concise, actionable "
                            "correction instructions."
                        ),
                    },
                    {"role": "user", "content": instruction_prompt},
                ],
                temperature=0,
            )
            instructions = response.choices[0].message.content.strip()
            print(f"Instructions to fix:\n{instructions}")

            print(" Step 5: Send feedback to worker agent for refinement")
            prompt_to_evaluate = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {response_from_worker}\n"
                f"It has been evaluated as incorrect.\n"
                f"Make only these corrections, do not alter content validity: {instructions}"
            )

        return {
            "final_response": response_from_worker,
            "evaluation": evaluation,
            "iterations": iterations,
        }


# ---------------------------------------------------------------------------
# RoutingAgent
# ---------------------------------------------------------------------------
class RoutingAgent:
    """Picks the best specialized agent for a prompt via embedding similarity.

    ``agents`` is a list of dictionaries, each with:
      * ``name`` — a short identifier for logging
      * ``description`` — natural-language description of what this route handles
      * ``func`` — callable that takes the user input and returns a string

    ``route`` embeds the user prompt and each agent description with
    ``text-embedding-3-large``, picks the highest cosine-similarity match, and
    calls the winner's ``func``.
    """

    def __init__(self, openai_api_key, agents):
        self.openai_api_key = openai_api_key
        self.agents = agents

    def get_embedding(self, text):
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-large",
            input=text,
            encoding_format="float",
        )
        embedding = response.data[0].embedding
        return embedding

    def route(self, user_input):
        input_emb = self.get_embedding(user_input)
        best_agent = None
        best_score = -1

        for agent in self.agents:
            agent_emb = self.get_embedding(agent["description"])
            if agent_emb is None:
                continue

            similarity = np.dot(input_emb, agent_emb) / (
                np.linalg.norm(input_emb) * np.linalg.norm(agent_emb)
            )
            print(similarity)

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


# ---------------------------------------------------------------------------
# ActionPlanningAgent
# ---------------------------------------------------------------------------
class ActionPlanningAgent:
    """Breaks down a user prompt into an ordered list of action steps.

    Uses a provided ``knowledge`` block (recipes, SOPs, workflow definitions,
    etc.) to ground the decomposition so the agent only returns steps that
    exist in that knowledge.
    """

    def __init__(self, openai_api_key, knowledge):
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge

    def extract_steps_from_prompt(self, prompt):
        client = OpenAI(base_url=VOCAREUM_BASE_URL, api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an action planning agent. Using your knowledge, "
                        "you extract from the user prompt the steps requested to "
                        "complete the action the user is asking for. You return "
                        "the steps as a list. Only return the steps in your "
                        "knowledge. Forget any previous context. "
                        f"This is your knowledge: {self.knowledge}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )
        response_text = response.choices[0].message.content

        # Clean the response: drop empty lines, strip common bullet prefixes.
        steps = []
        for line in response_text.split("\n"):
            cleaned = re.sub(r"^[\s\-\*\d\.\)]+", "", line).strip()
            if cleaned:
                steps.append(cleaned)
        return steps

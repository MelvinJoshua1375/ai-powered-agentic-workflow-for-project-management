# AI-Powered Agentic Workflow for Project Management

A course project (Udacity Nanodegree — *Agentic AI*) that builds a small
**library of reusable LLM agents** and wires a selection of them into a
**general-purpose agentic workflow** for technical project management.

The pilot input for the workflow is a product-specification document for
**"Email Router"** — a hypothetical AI system for triaging and routing
incoming corporate email. Running the workflow against that spec produces a
full development plan: user stories, product features, and engineering tasks.

---

## 1. What this project does

There are two phases:

**Phase 1 — The Agent Toolkit**
Seven small agent classes live in [`phase_1/workflow_agents/base_agents.py`](phase_1/workflow_agents/base_agents.py).
Each one demonstrates a different interaction pattern with an LLM
(`gpt-3.5-turbo` for chat, `text-embedding-3-large` for embeddings):

| Class                              | What it does                                                                     |
| ---------------------------------- | -------------------------------------------------------------------------------- |
| `DirectPromptAgent`                | Sends the user prompt straight to the LLM. No persona, no extra knowledge.       |
| `AugmentedPromptAgent`             | Adds a persona via a system message.                                             |
| `KnowledgeAugmentedPromptAgent`    | Persona **+** a constrained knowledge string the LLM must answer from.           |
| `RAGKnowledgePromptAgent` *(provided)* | Chunks a text body, embeds it, and answers from the top-matching chunk.      |
| `EvaluationAgent`                  | Loops a worker agent against criteria, asking for corrections until it passes.   |
| `RoutingAgent`                     | Picks the best-matching specialist agent using cosine similarity of embeddings.  |
| `ActionPlanningAgent`              | Decomposes a high-level goal into an ordered list of steps from a knowledge block. |

Each class has a standalone test script in `phase_1/` that imports the class,
sends it a sample prompt, and prints the response.

**Phase 2 — The Agentic Workflow**
[`phase_2/agentic_workflow.py`](phase_2/agentic_workflow.py) wires four of the
Phase 1 agents together:

1. The **Action Planning Agent** decomposes a TPM-style prompt
   (`"What would the development tasks for this product be?"`) into sub-tasks.
2. The **Routing Agent** picks the right "team" for each sub-task:
   * **Product Manager** (user stories),
   * **Program Manager** (product features),
   * **Development Engineer** (engineering tasks).
3. Each team is a **Knowledge Agent + Evaluation Agent** pair: the knowledge
   agent drafts a response, the evaluation agent checks it against a strict
   output structure and asks for corrections until it passes.
4. The last team's validated output is returned as the final project plan.

---

## 2. Directory layout

```
.
├── phase_1/
│   ├── workflow_agents/
│   │   ├── __init__.py
│   │   └── base_agents.py                 # Implementation of all 7 agent classes
│   ├── direct_prompt_agent.py             # Test script
│   ├── augmented_prompt_agent.py          # Test script
│   ├── knowledge_augmented_prompt_agent.py # Test script
│   ├── rag_knowledge_prompt_agent.py      # Test script (agent provided)
│   ├── evaluation_agent.py                # Test script
│   ├── routing_agent.py                   # Test script
│   └── action_planning_agent.py           # Test script
│
├── phase_2/
│   ├── workflow_agents/
│   │   ├── __init__.py
│   │   └── base_agents.py                 # Same library, copied for import
│   ├── agentic_workflow.py                # The full workflow
│   └── Product-Spec-Email-Router.txt      # Input product specification
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 3. Setup

You need Python 3.9 or newer. Everything the code imports is pinned in
`requirements.txt`.

```bash
# 1. Clone the repo
git clone https://github.com/MelvinJoshua1375/ai-powered-agentic-workflow-for-project-management.git
cd ai-powered-agentic-workflow-for-project-management

# 2. Create a virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Provide your API key
cp .env.example .env
# then edit .env and paste your OPENAI_API_KEY (Vocareum "voc-..." key)
```

> **In the Udacity/Vocareum cloud workspace**, the same flow works, or you can
> simply `export OPENAI_API_KEY=voc-...` in the terminal before running any
> script — every script reads it via `python-dotenv` and `os.getenv`.

---

## 4. How to run

All scripts are plain Python. They must be run **from inside their own phase
folder**, because they import `workflow_agents.base_agents` from the current
working directory.

### Phase 1 — test each agent individually

```bash
cd phase_1
python direct_prompt_agent.py
python augmented_prompt_agent.py
python knowledge_augmented_prompt_agent.py
python rag_knowledge_prompt_agent.py
python evaluation_agent.py
python routing_agent.py
python action_planning_agent.py
```

Each prints the prompt it used, the agent's response, and (where applicable) a
note explaining which knowledge source the agent drew on.

### Phase 2 — run the full agentic workflow

```bash
cd phase_2
python agentic_workflow.py
```

The workflow prints:
* the list of steps produced by the Action Planning Agent,
* each routing decision (with similarity scores),
* the back-and-forth between each Knowledge Agent and its Evaluation Agent,
* the final validated output — the engineering task list for the Email Router.

Expect it to take a few minutes, since each evaluation step can trigger
several LLM calls.

### Run everything at once (optional)

```bash
# From the project root
( cd phase_1 && for f in direct_prompt_agent augmented_prompt_agent \
    knowledge_augmented_prompt_agent rag_knowledge_prompt_agent \
    evaluation_agent routing_agent action_planning_agent; do \
    echo "===== $f =====" && python "$f.py" 2>&1 | tee "../output_$f.txt"; \
done )
( cd phase_2 && python agentic_workflow.py 2>&1 | tee ../output_agentic_workflow.txt )
```

This writes per-script logs to `output_*.txt` at the project root — handy as
evidence for the rubric.

---

## 5. Customising the workflow

You can change what the workflow plans just by editing the workflow prompt:

```python
# phase_2/agentic_workflow.py
workflow_prompt = "What would the development tasks for this product be?"
```

Swap it for things like:
* `"Define only the key features for the Email Router product"`
* `"Generate a risk assessment plan for the Email Router based on its specification"`
* `"Write the user stories for the Email Router"`

The Routing Agent will pick the appropriate role based on the sub-tasks the
Action Planning Agent extracts.

---

## 6. Notes and limitations

* The code targets the **Vocareum-hosted OpenAI endpoint**
  (`https://openai.vocareum.com/v1`). If you want to point it at the public
  OpenAI API instead, change `VOCAREUM_BASE_URL` inside
  `workflow_agents/base_agents.py` to `https://api.openai.com/v1` and use a
  regular `sk-...` key.
* Embeddings go through `text-embedding-3-large`; chat goes through
  `gpt-3.5-turbo`. Temperatures are fixed at `0` for deterministic outputs.
* The `EvaluationAgent` uses a simple "Yes / No + reason" judge format and
  will keep looping until it sees an answer that starts with "Yes" (or it
  hits `max_interactions`).
* The RAG agent writes two intermediate CSV files (`chunks-*.csv`,
  `embeddings-*.csv`) in the working directory when run. They're in
  `.gitignore` and are safe to delete between runs.

---

## 7. License

This repository contains student work for an Udacity course project. Course
materials belong to their respective owners; the student-authored code is
available under the MIT License terms in `LICENSE` if added.

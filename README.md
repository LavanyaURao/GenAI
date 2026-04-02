# Multi-Agent RAG System for Hallucination Reduction

## Problem Statement
Large Language Models continue to produce inaccurate responses due to hallucinations, particularly during complex reasoning, even when supported by Retrieval-Augmented Generation. While existing methods improve factual grounding, they fail to effectively validate reasoning consistency and evidence-based conclusions.

## Abstract
Large Language Models (LLMs) often produce fluent but incorrect responses due to hallucinations, especially when reasoning beyond retrieved facts. While RAG reduces factual errors by grounding responses in external documents, it does not adequately address reasoning-level hallucinations. This project proposes a lightweight multi-agent framework in which specialized LLM agents perform answer generation, external critique, and final judgment. By enforcing evidence-based reasoning and explicit rejection of unsupported claims, the system significantly reduces hallucinations compared to self-verification approaches.

---

## Models Used

| Component | Model | Provider |
|---|---|---|
| Embedding / Retrieval | `all-MiniLM-L6-v2` | HuggingFace (local) |
| Generator Agent | `llama3-8b-8192` (Llama 3 8B) | Groq API |
| Critic Agent | `llama3-8b-8192` (Llama 3 8B) | Groq API |
| Judge Agent | `llama3-8b-8192` (Llama 3 8B) | Groq API |

---

## Project Structure

```
GenAI/
│
├── agents/
│   ├── critic.py                # Critic Agent
│   ├── generator.py             # Generator Agent
│   ├── judge.py                 # Judge Agent
│   └── retriever.py             # Retriever Agent
│
├── data/
│   └── knowledge_base.txt       # Domain knowledge (10 topics)
│
├── database/
│   └── vector_store.py          # FAISS + sentence-transformers
│
├── pipeline/
│   └── multi_agent_pipeline.py  # Orchestrates all agents
│
├── main.py                      # Entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup & Run

### 1. Get a FREE Groq API Key
Go to [console.groq.com](https://console.groq.com) → API Keys → Create Key

### 2. Configure `.env`
```bash
cp .env.example .env
# Open .env and paste your key:
# GROQ_API_KEY=your_key_here
```

### 3. Create virtual environment
```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run
```bash
python main.py
```

Results saved to `results.json`.

---

## Pipeline Flow

```
User Query
    ↓
[RetrieverAgent]  → FAISS semantic search → top-3 chunks
    ↓
[GeneratorAgent]  → Llama 3 8B → initial answer (context-only)
    ↓
[CriticAgent]     → Llama 3 8B → flags unsupported claims
    ↓
[JudgeAgent]      → Llama 3 8B → final corrected answer
    ↓
Final Answer (hallucination-reduced)
```

---

## References
1. Lewis et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*
2. Meta AI. *Llama 3 Model Card*
3. Reimers & Gurevych (2019). *Sentence-BERT*
4. Johnson et al. (2017). *FAISS: Billion-scale similarity search with GPUs*

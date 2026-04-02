"""
multi_agent_pipeline.py
-----------------------
LLM-ONLY Multi-Agent Pipeline:
  1. GeneratorAgent → produces initial answer
  2. CriticAgent   → checks correctness
  3. JudgeAgent    → refines final answer

(NO RAG / NO FAISS / NO VECTOR STORE)
"""

from groq import Groq

from agents.generator import GeneratorAgent
from agents.critic import CriticAgent
from agents.judge import JudgeAgent


class MultiAgentPipeline:
    def __init__(self, api_key: str):
        print("=" * 60)
        print("  Multi-Agent LLM Pipeline (Groq + Llama 3)")
        print("=" * 60)

        # Initialize Groq client
        self.client = Groq(api_key=api_key)

        # Initialize agents
        self.generator = GeneratorAgent(self.client)
        self.critic = CriticAgent(self.client)
        self.judge = JudgeAgent(self.client)

        print("Pipeline ready.\n")

    def run(self, query: str) -> dict:
        print("=" * 60)
        print(f"Query: {query}")
        print("=" * 60)

        # Step 1: Generate
        initial_answer = self.generator.generate(query)

        # Step 2: Critique
        critique = self.critic.critique(query, initial_answer)

        # Step 3: Judge
        final_answer = self.judge.judge(query, initial_answer, critique)

        result = {
            "query": query,
            "initial_answer": initial_answer,
            "critique": critique,
            "final_answer": final_answer,
        }

        self._print_result(result)
        return result

    def _print_result(self, result: dict):
        print("\n" + "=" * 60)
        print("PIPELINE RESULTS")
        print("=" * 60)
        print(f"\n[QUERY]\n{result['query']}\n")
        print(f"[INITIAL ANSWER]\n{result['initial_answer']}\n")
        print(f"[CRITIC FEEDBACK]\n{result['critique']}\n")
        print(f"[FINAL ANSWER]\n{result['final_answer']}\n")
        print("=" * 60)
"""
generator.py
------------
Generator Agent: Uses Groq (Llama 3) to produce an answer
directly using the LLM's knowledge (NO RAG / NO CONTEXT).
"""

from groq import Groq


class GeneratorAgent:
    def __init__(self, client: Groq):
        self.client = client
        self.model = "llama-3.3-70b-versatile"

    def generate(self, query: str) -> str:
        """
        Generate an answer using ONLY the LLM (no context).

        Args:
            query: The user's question.

        Returns:
            Generated answer as a string.
        """
        print("[GeneratorAgent] Generating answer using LLM...")

        system_prompt = (
            "You are a helpful, knowledgeable, and accurate AI assistant. "
            "Answer the user's question clearly and concisely. "
            "You may use your general knowledge to provide the best possible answer."
            
        )

        user_prompt = f"""Question: {query}

Answer:"""

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.choices[0].message.content.strip()
        print("[GeneratorAgent] Answer generated.")
        return answer
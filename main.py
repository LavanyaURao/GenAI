"""
main.py
-------
Interactive Multi-Agent LLM system (NO RAG).

Allows user to input queries dynamically.
Saves results in clean structured JSON format.

Type 'exit' to quit.
"""

import os
import json
from dotenv import load_dotenv
from pipeline.multi_agent_pipeline import MultiAgentPipeline

# Load API key from .env
load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")

if API_KEY == "your_groq_api_key_here":
    print("❌ ERROR: Please set your GROQ_API_KEY in the .env file.")
    print("Get a free key at https://console.groq.com")
    exit(1)


def main():
    print("=" * 60)
    print("🤖 Multi-Agent LLM Chat (Groq + Llama 3)")
    print("Type 'exit' to quit")
    print("=" * 60)

    # Initialize pipeline
    pipeline = MultiAgentPipeline(api_key=API_KEY)

    all_results = []

    while True:
        query = input("\n🧑 You: ")

        if query.lower() in ["exit", "quit"]:
            print("\n👋 Exiting... Bye!")
            break

        if not query.strip():
            print("⚠️ Please enter a valid question.")
            continue

        # Run pipeline
        result = pipeline.run(query)

        # ✅ Format result (THIS is the key change)
        formatted_result = {
            "query": result["query"],
            "initial_answer": result["initial_answer"],
            "critique": result["critique"],
            "final_answer": result["final_answer"]
        }

        all_results.append(formatted_result)

        # Print final answer nicely
        print("\n🤖 Final Answer:")
        print(formatted_result["final_answer"])

    # Save all conversation results
    output_path = "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\n✅ All chat results saved to {output_path}")


if __name__ == "__main__":
    main()
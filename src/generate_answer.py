"""
End-to-end RAG: hybrid retrieval (Phase 5) + LLM generation with citations,
using a local model via Ollama - no API key, no cost, runs on your machine.

Setup (one-time):
    1. Install Ollama: https://ollama.com/download
    2. Pull a model:   ollama pull llama3.1:8b
    3. Make sure Ollama is running (it starts automatically after install,
       or run `ollama serve` manually)

Usage:
    python src/generate_answer.py "what is the maximum output current of the LM317"
    python src/generate_answer.py "TPS62840 quiescent current" --model llama3.1:8b --top-k 5
"""

import argparse
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import ollama

from hybrid_query import load_retrieval_backend, retrieve_chunks

DEFAULT_MODEL = "llama3.1:8b"

# The instruction discipline here is the whole point of RAG over a
# semiconductor datasheet: the model must NEVER state a specific number,
# voltage, current, or rating from its own training data - only from the
# retrieved excerpts - and must say so plainly when the excerpts don't
# cover the question, rather than guessing at a plausible-sounding value.
SYSTEM_PROMPT = """You are a semiconductor datasheet assistant. You answer questions ONLY using the excerpts provided below - never from your own general knowledge of these parts.

Rules:
- Every specific number, voltage, current, temperature, or rating you state MUST come from the excerpts, and MUST be followed by a citation like [Source 2].
- If you find the answer in the excerpts, provide it directly. Do NOT start your response with "The provided excerpts don't cover this" if you are going to provide the answer anyway. Only say "The provided excerpts don't cover this" if you are completely unable to answer the question.
- Do NOT guess or fill in a plausible-sounding value from training data.
- Often, datasheets specify different values for different conditions (e.g., Min/Typ/Max values, different packages like PWP vs RSA, or different modes like Active vs Sleep). These are NOT "discrepancies" or contradictions. Do not state there is a discrepancy in the datasheet; instead, list the values alongside their specific conditions or modes.
- Keep the answer concise and direct - this is for an engineer who wants the spec, not a lecture."""


def build_context(results: list) -> str:
    """Format retrieved chunks into a numbered context block the model can
    cite back to by number."""
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[Source {i}] Part: {r['part_number']} | Section: {r['section']} | Page: {r['page_number']}\n"
            f"{r['content']}\n"
        )
    return "\n".join(lines)


def generate_answer(question: str, model_name: str = DEFAULT_MODEL, top_k: int = 5):
    print("Loading retrieval backend...")
    backend = load_retrieval_backend()

    print(f"Retrieving context for: {question}\n")
    outcome = retrieve_chunks(backend, question, top_k)
    results = outcome["results"]

    if not results:
        print("No relevant chunks found in the vector store.")
        return

    context = build_context(results)
    user_message = f"Excerpts from semiconductor datasheets:\n\n{context}\n\nQuestion: {question}"

    print(f"Generating answer with {model_name}...\n")
    try:
        response = ollama.chat(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
    except Exception as e:
        print(f"Could not reach Ollama or run model '{model_name}': {e}")
        print("Make sure Ollama is installed and running, and the model is pulled:")
        print(f"  ollama pull {model_name}")
        return

    answer = response["message"]["content"]

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer)

    print("\n" + "=" * 70)
    print("SOURCES")
    print("=" * 70)
    for i, r in enumerate(results, start=1):
        print(f"[Source {i}] {r['part_number']} - {r['section']} (page {r['page_number']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG: retrieve + generate a cited answer from datasheet excerpts")
    parser.add_argument("question", help="Question to answer")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Ollama model name (default: llama3.1:8b)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve as context")
    args = parser.parse_args()

    generate_answer(args.question, model_name=args.model, top_k=args.top_k)
"""
Lesson 4: reranking. Retrieve many, reorder carefully, keep a few.

Search is fast because it compares vectors and words, not meaning in depth.
A reranker reads the question and each candidate TOGETHER and scores how well
the candidate answers it. Slower and costlier, so only for the top 10-50.

    Part 1  an LLM reranker: the chat model scores 10 hybrid candidates 0-3.
            Works on any tier, costs one chat call per question.
    Part 2  the Azure AI Search semantic ranker: the same idea as a built-in feature
            (reranker score 0-4). It needs the Basic tier or higher; on the class's
            Free tier this part prints why it is not available.

Run:  python 04_reranking.py
      python 04_reranking.py "your question"
"""
import json
import sys

from azure.core.exceptions import HttpResponseError

from azure_client import chat
from rag import Match, search, search_error_hint, show_matches

CANDIDATES = 10
KEEP = 3

RERANK_SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"source": {"type": "integer"}, "relevance": {"type": "integer", "enum": [0, 1, 2, 3]}},
                "required": ["source", "relevance"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["scores"],
    "additionalProperties": False,
}


def llm_rerank(question: str, matches: list[Match]) -> list[tuple[int, Match]]:
    """Ask the chat model for a relevance score per candidate, then sort by it (ties keep the search order)."""
    numbered = "\n\n".join(f"[{n}] {m.content}" for n, m in enumerate(matches, start=1))
    reply = chat([
        {"role": "system", "content": "Score how well each numbered text answers the question. 3 = answers it, "
                                      "2 = partly, 1 = same topic only, 0 = unrelated. Score every text."},
        {"role": "user", "content": f"Question: {question}\n\nTexts:\n{numbered}"},
    ], response_format={"type": "json_schema", "json_schema": {"name": "rerank", "strict": True, "schema": RERANK_SCHEMA}})
    relevance = {item["source"]: item["relevance"] for item in json.loads(reply)["scores"]}
    scored = [(relevance.get(n, 0), m) for n, m in enumerate(matches, start=1)]
    return sorted(scored, key=lambda pair: pair[0], reverse=True)              # sorted() is stable


def demo(question: str) -> None:
    print(f"\nQuestion: {question}")
    matches = search(question, top=CANDIDATES)
    print(f" Hybrid search, top {KEEP} of {CANDIDATES}:")
    show_matches(matches[:KEEP])

    print(f" After the LLM reranker (relevance 0-3), top {KEEP}:")
    for relevance, m in llm_rerank(question, matches)[:KEEP]:
        print(f"  relevance {relevance}  (was rank {matches.index(m) + 1:>2})  {m.title}")

    print(" After the Azure semantic ranker:")
    try:
        show_matches(search(question, top=KEEP, semantic=True))
    except HttpResponseError as error:
        print(f"  Not available: {search_error_hint(error)}")


if __name__ == "__main__":
    questions = sys.argv[1:] or [
        "Is my phone covered if I drop it and the screen cracks?",
        "What is the capital of France?",
    ]
    for question in questions:
        demo(question)
    print("\nSearch ALWAYS returns something, even for France. A reranker score (LLM 0-3, semantic 0-4)")
    print("says how relevant the best chunk really is: a good signal for 'I don't know' (lesson 5).")

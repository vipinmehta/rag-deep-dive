"""
Lesson 7: tune retrieval with numbers. Chunking × search mode × top-k.

Lesson 6 measured one configuration end to end. Most RAG problems are retrieval
problems, and retrieval is cheap to measure: no chat model, one search per
question. This lesson runs the answerable questions of the test set against
every chunking strategy in every search mode, and prints one table.

One search with top 5 gives hit@1, hit@3 and hit@5 at once: the first 1, 3 or 5 results.

Run:  python 07_tune.py
      python 07_tune.py --semantic     (adds hybrid + semantic ranker: Basic tier or higher)
"""
import sys

from azure.core.exceptions import HttpResponseError

from azure_client import embed
from rag import MODES, STRATEGIES, TEST_SET, fact_found, rank_of, search, search_error_hint

cases = [case for case in TEST_SET if case.source]
vectors = embed([case.question for case in cases])                     # embed the questions once, reuse everywhere
configs = [(strategy, mode, False) for strategy in STRATEGIES for mode in MODES]
if "--semantic" in sys.argv:
    configs += [(strategy, "hybrid", True) for strategy in STRATEGIES]

print(f"{len(cases)} questions, {len(configs)} configurations, {len(cases) * len(configs)} searches, no chat calls.\n")
print(f"  {'Chunking':<10} {'Mode':<16} {'hit@1':>6} {'hit@3':>6} {'hit@5':>6} {'fact@3':>7} {'MRR':>5}")
results = []
for strategy, mode, semantic in configs:
    ranks, fact_hits = [], 0
    try:
        for case, vector in zip(cases, vectors):
            matches = search(case.question, mode=mode, top=5, chunking=strategy, semantic=semantic, vector=vector)
            ranks.append(rank_of(case, [m.source for m in matches]))
            fact_hits += fact_found(case, [m.content for m in matches[:3]])
    except HttpResponseError as error:
        print(f"  {strategy:<10} {'hybrid+semantic':<16} not available: {search_error_hint(error)}")
        continue
    hit = {k: sum(1 for r in ranks if 0 < r <= k) for k in (1, 3, 5)}
    mrr = sum(1 / r for r in ranks if r) / len(cases)
    name = mode + ("+semantic" if semantic else "")
    results.append((mrr, strategy, name))
    print(f"  {strategy:<10} {name:<16} {hit[1]:>3}/{len(cases)} {hit[3]:>3}/{len(cases)} {hit[5]:>3}/{len(cases)}"
          f" {fact_hits:>4}/{len(cases)} {mrr:>5.2f}")

if results:
    mrr, strategy, mode = max(results, key=lambda result: result[0])          # the first of any ties
    print(f"\nBest on this test set: {strategy} chunks, {mode} search (MRR {mrr:.2f}).")
print("Eleven questions is a tiny test set: one question is 9 points. Before you trust a winner, add questions")
print("from real users, especially the ones that FAILED, and run the table again.")

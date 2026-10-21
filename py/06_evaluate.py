"""
Lesson 6: evaluate retrieval and answers on a small test set.

"It answered my three questions" is not a test. Fourteen questions with known
answers (TEST_SET in rag.py) tell you whether a change made things better or worse.

    retrieval   hit rate@3   the right file is in the top 3
                fact@3       the fact itself is in the 3 chunks (a badly cut chunk can miss it)
                MRR          mean of 1/rank of the right file: 1.0 = always first
    answers     correct      contains the expected fact, or says NO_ANSWER when it should
    grounding   an LLM judge checks every claim in the answer against the sources
                (structured output, so code can count the verdicts)

Run:  python 06_evaluate.py
      python 06_evaluate.py --semantic     (with the semantic ranker: Basic tier or higher)
"""
import json
import sys

from azure_client import chat
from rag import MIN_SIMILARITY, NO_ANSWER, TEST_SET, Match, fact_found, format_sources, generate, rank_of, search

TOP_K = 3
SEMANTIC = "--semantic" in sys.argv

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["grounded", "unsupported_claims"],
    "additionalProperties": False,
}


def judge(answer: str, matches: list[Match]) -> dict:
    """LLM-as-judge: is every claim in the answer stated in the sources?"""
    reply = chat([
        {"role": "system", "content": "You check a support answer against its sources. grounded = true only if every "
                                      "factual claim in the answer is stated in the sources. Citation numbers like [1] "
                                      f"are not claims. The reply '{NO_ANSWER}' is always grounded. "
                                      "List each unsupported claim in a few words."},
        {"role": "user", "content": f"Sources:\n{format_sources(matches)}\n\nAnswer:\n{answer}"},
    ], response_format={"type": "json_schema", "json_schema": {"name": "verdict", "strict": True, "schema": JUDGE_SCHEMA}})
    return json.loads(reply)


hits = facts = correct = grounded = 0
reciprocal_ranks = []
print(f"Hybrid search{' + semantic ranker' if SEMANTIC else ''}, heading chunks, top {TOP_K}\n")
for case in TEST_SET:
    matches = search(case.question, top=TOP_K, semantic=SEMANTIC)
    best = max((m.similarity or 0 for m in matches), default=0)
    answer = NO_ANSWER if best < MIN_SIMILARITY else generate(case.question, matches)

    if case.source:                                                    # retrieval: only for answerable questions
        rank = rank_of(case, [m.source for m in matches])
        hits += rank > 0
        facts += fact_found(case, [m.content for m in matches])
        reciprocal_ranks.append(1 / rank if rank else 0)
        ok = any(fact.lower() in answer.lower() for fact in case.facts)
        retrieved = f"rank {rank}" if rank else "MISSED"
    else:
        ok = NO_ANSWER.lower() in answer.lower()
        retrieved = "no answer"
    verdict = judge(answer, matches) if answer != NO_ANSWER else {"grounded": True, "unsupported_claims": []}
    correct += ok
    grounded += verdict["grounded"]

    print(f"  {'PASS' if ok else 'FAIL'}  {retrieved:<9} {'grounded' if verdict['grounded'] else 'NOT GROUNDED':<12} {case.question}")
    if not ok:
        print(f"        expected {case.facts or [NO_ANSWER]}, got: {answer[:100]}")
    for claim in verdict["unsupported_claims"]:
        print(f"        unsupported: {claim}")

answerable = sum(1 for case in TEST_SET if case.source)
print()
print(f"  Retrieval   hit rate@{TOP_K} {hits}/{answerable}   fact@{TOP_K} {facts}/{answerable}   "
      f"MRR {sum(reciprocal_ranks) / answerable:.2f}")
print(f"  Answers     correct {correct}/{len(TEST_SET)}")
print(f"  Grounding   judged grounded {grounded}/{len(TEST_SET)}")
print("\nWrite these numbers down. Change ONE thing (top_k, chunking, prompt, gate), run again, compare.")
print("A judge is a model too: read the FAILs and the unsupported claims yourself before you trust the totals.")

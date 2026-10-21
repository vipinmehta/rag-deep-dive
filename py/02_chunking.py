"""
Lesson 2: chunking. How you cut the documents decides what retrieval can find.

Three common strategies (the code is in rag.py, about ten lines each):
    fixed       every N characters with some overlap; ignores the structure
    paragraph   one chunk per paragraph
    heading     one chunk per '## ' section, the heading kept in the title

Too small: a chunk loses its context ("It costs 8%" - what does?).
Too big:   one chunk covers many topics, its embedding gets blurry, and every
           question sends the model more text than it needs (tokens = money).

This lesson searches in memory with cosine similarity, like the Ollama day, so
you can try any setting without rebuilding an index.

Run:  python 02_chunking.py
"""
from azure_client import embed
from rag import TEST_SET, chunk_all, cosine_similarity, fact_found, rank_of

print("=== 1. The same seven documents, cut three ways ===")
for strategy in ["fixed", "paragraph", "heading"]:
    chunks = chunk_all(strategy)
    lengths = [len(c.content) for c in chunks]
    print(f"  {strategy:<10} {len(chunks):>3} chunks, average {sum(lengths) // len(lengths):>4} characters, longest {max(lengths):>5}")

print()
print("=== 2. Where the fact about ShopEase Protect's price ends up ===")
for strategy in ["fixed", "paragraph", "heading"]:
    chunk = next(c for c in chunk_all(strategy) if "8% of" in c.content)
    body = chunk.content[len(chunk.title) + 2:]
    at = body.index("8% of")
    print(f"  [{strategy}] {chunk.id}, {len(chunk.content)} characters, title '{chunk.title}'")
    print(f"    starts: {body[:50]}...")
    print(f"    fact:   ...{body[max(0, at - 70):at + 30]}...")
print("  The fixed chunk starts mid-sentence: does it still say WHAT costs 8%? Only the heading chunk's title does.")

print()
print("=== 3. Where fixed-size cuts fall (300 characters, 50 overlap) ===")
for chunk in [c for c in chunk_all("fixed") if c.source == "warranty-and-repairs.md"][:3]:
    body = chunk.content[len(chunk.title) + 2:]
    print(f"  starts '{body[:28]}'  ends '{body[-28:]}'")
print("  The overlap repeats the last 50 characters of each chunk at the start of the next one.")

print()
print("=== 4. Does retrieval get better or worse? ===")
CONFIGS = [
    ("fixed 100, no overlap", "fixed", {"size": 100, "overlap": 0}),
    ("fixed 300, no overlap", "fixed", {"size": 300, "overlap": 0}),
    ("fixed 300, overlap 50", "fixed", {"size": 300, "overlap": 50}),
    ("fixed 1000, overlap 100", "fixed", {"size": 1000, "overlap": 100}),
    ("paragraph", "paragraph", {}),
    ("heading", "heading", {}),
]
cases = [case for case in TEST_SET if case.source]                    # the questions that have an answer
question_vectors = embed([case.question for case in cases])           # embed the questions once
print(f"{len(cases)} questions, top 3 chunks each. 'File' = right document first; 'Fact' = the fact is in the 3 chunks.")
print(f"  {'Chunking':<24} {'Chunks':>6} {'Avg chars':>9} {'File@1':>7} {'Fact@3':>7}  {'Text sent per question':>22}")
for name, strategy, options in CONFIGS:
    chunks = chunk_all(strategy, **options)
    vectors = embed([c.content for c in chunks])                      # one request per configuration
    file_hits = fact_hits = sent = 0
    for case, question_vector in zip(cases, question_vectors):
        ranked = sorted(zip(chunks, vectors), key=lambda pair: cosine_similarity(question_vector, pair[1]), reverse=True)
        top = [chunk for chunk, _ in ranked[:3]]
        file_hits += rank_of(case, [c.source for c in top]) == 1
        fact_hits += fact_found(case, [c.content for c in top])
        sent += sum(len(c.content) for c in top)
    average = sum(len(c.content) for c in chunks) // len(chunks)
    print(f"  {name:<24} {len(chunks):>6} {average:>9} {file_hits:>4}/{len(cases)} {fact_hits:>4}/{len(cases)}"
          f"  {sent // len(cases):>14} chars ≈ {sent // len(cases) // 4} tokens")

print()
print("Read the table: tiny chunks lose facts at their edges, huge chunks blur the meaning and cost more tokens.")
print("Cut where the author changed topic, keep the title in every chunk, and add overlap when you must cut by size.")

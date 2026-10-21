"""
Lesson 1: embeddings, meaning as numbers.

An embedding model turns a text into a list of numbers (a vector). Texts with a
similar meaning get vectors that point in a similar direction, even when they
share no words. Cosine similarity measures that direction:

    cos(a, b) = (a · b) / (|a| × |b|)          1 = same meaning, 0 = unrelated

On the Ollama day nomic-embed-text gave 768 numbers. The class's 'embed'
deployment (text-embedding-3-small) gives 1,536, and can give fewer if you ask.

Run:  python 01_embeddings.py
"""
import math

from azure_client import EMBED_DEPLOYMENT, client, embed
from rag import cosine_similarity

print("=== 1. One sentence becomes 1,536 numbers ===")
reply = client.embeddings.create(model=EMBED_DEPLOYMENT, input=["My parcel is late."])
vector = reply.data[0].embedding
print(f"Model:      {reply.model}")
print(f"Dimensions: {len(vector)}")
print(f"First 6:    {[round(x, 4) for x in vector[:6]]} ...")
print(f"Length:     {math.sqrt(sum(x * x for x in vector)):.4f}   (always 1: the vector is normalised, so only its direction counts)")
print(f"Tokens:     {reply.usage.prompt_tokens}   (embeddings bill input tokens only: about USD 0.02 per million)")

print()
print("=== 2. Similar vs unrelated sentences ===")
ANCHOR = "My parcel is late."
OTHERS = [
    ("My order still has not arrived.", "same meaning, no shared words"),
    ("Where is my delivery?", "same topic, a question"),
    ("मेरा पार्सल देर से आया है।", "same meaning, in Hindi"),
    ("My parcel is lovely.", "shared words, different meaning"),
    ("How do I reset my password?", "ShopEase, but another topic"),
    ("The cat sat on the mat.", "unrelated"),
]
vectors = embed([ANCHOR] + [sentence for sentence, _ in OTHERS])          # one request for all seven
print(f"Compared with: '{ANCHOR}'")
for (sentence, kind), other in zip(OTHERS, vectors[1:]):
    score = cosine_similarity(vectors[0], other)
    print(f"  {score:.3f}  {'#' * round(score * 30):<30}  {sentence:<34} ({kind})")
print("Read the gaps, not the absolute values: with this model even 'unrelated' rarely drops below 0.05.")

print()
print("=== 3. Fewer dimensions: smaller vectors, almost the same ranking ===")
short = embed([ANCHOR] + [sentence for sentence, _ in OTHERS], dimensions=256)
print(f"{'1,536 dims':>10}  {'256 dims':>8}   sentence")
for (sentence, _), full_vector, short_vector in zip(OTHERS, vectors[1:], short[1:]):
    print(f"{cosine_similarity(vectors[0], full_vector):>10.3f}  {cosine_similarity(short[0], short_vector):>8.3f}   {sentence}")
print(f"Storage per chunk: 1,536 × 4 bytes = {1536 * 4:,} bytes, 256 × 4 bytes = {256 * 4:,} bytes.")
print("The index must use the SAME model and the SAME dimensions for the chunks and for every question.")

print()
print("=== 4. Cosine similarity is just a dot product here ===")
dot = sum(x * y for x, y in zip(vectors[0], vectors[1]))
print(f"dot product {dot:.6f}  =  cosine {cosine_similarity(vectors[0], vectors[1]):.6f}   (both vectors have length 1)")

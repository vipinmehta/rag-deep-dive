"""
Lesson 3: retrieval with Azure AI Search. Keyword vs vector vs hybrid, and top-k.

    keyword   BM25 on the words. Great for codes, names and numbers (FESTIVE10, 1800-200-7327).
              Misses synonyms: "fridge" does not match "refrigerator".
    vector    nearest embeddings. Finds the meaning in other words ("parcel stuck" -> "Held at hub").
              Can miss exact codes and rare words.
    hybrid    both at once; Azure AI Search merges the two ranked lists with
              Reciprocal Rank Fusion (RRF): score = sum of 1 / (60 + rank) over both lists.

In code the three modes differ in two arguments (see search() in rag.py):
    keyword   search_text=question
    vector    vector_queries=[VectorizedQuery(vector=..., k_nearest_neighbors=top, fields="vector")]
    hybrid    both

The shared index 'shopease-rag' holds the chunks of ALL three chunking strategies
from lesson 2, with a 'chunking' field to filter on. The trainer builds it once.

Run:  python 03_search_modes.py
      python 03_search_modes.py "your question"
      python 03_search_modes.py --build          (trainer only: needs AZURE_SEARCH_ADMIN_KEY)
"""
import sys
from dataclasses import asdict

from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration, SearchableField, SearchField, SearchFieldDataType, SearchIndex, SemanticConfiguration,
    SemanticField, SemanticPrioritizedFields, SemanticSearch, SimpleField, VectorSearch, VectorSearchProfile,
)

from azure_client import embed, setting
from rag import INDEX_NAME, MODES, STRATEGIES, chunk_all, search, show_matches


def build_index() -> None:
    """Trainer only: chunk with every strategy, embed, and upload to a fresh index. Safe to run twice."""
    endpoint, credential = setting("AZURE_SEARCH_ENDPOINT"), AzureKeyCredential(setting("AZURE_SEARCH_ADMIN_KEY"))
    chunks = [chunk for strategy in STRATEGIES for chunk in chunk_all(strategy)]
    vectors = embed([chunk.content for chunk in chunks])                       # one call for all chunks

    fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SimpleField(name="chunking", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SearchableField(name="title"),                                          # keyword search
        SearchableField(name="content"),                                        # keyword search
        SearchField(name="vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True, vector_search_dimensions=len(vectors[0]),
                    vector_search_profile_name="default"),                     # vector search
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],                   # cosine similarity by default
        profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")],
    )
    semantic = SemanticSearch(configurations=[SemanticConfiguration(            # for the semantic ranker (lesson 4)
        name="default",
        prioritized_fields=SemanticPrioritizedFields(title_field=SemanticField(field_name="title"),
                                                     content_fields=[SemanticField(field_name="content")]),
    )])

    indexes = SearchIndexClient(endpoint, credential)
    if INDEX_NAME in list(indexes.list_index_names()):
        indexes.delete_index(INDEX_NAME)                                        # start clean: old chunk ids go too
    try:
        indexes.create_index(SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search, semantic_search=semantic))
    except HttpResponseError as error:
        print(f"Could not add the semantic configuration ({str(error.message).splitlines()[0]}). Creating the index without it.")
        indexes.create_index(SearchIndex(name=INDEX_NAME, fields=fields, vector_search=vector_search))

    documents = [{**asdict(chunk), "vector": vector} for chunk, vector in zip(chunks, vectors)]
    results = SearchClient(endpoint, INDEX_NAME, credential).upload_documents(documents)
    counts = ", ".join(f"{s} {sum(c.chunking == s for c in chunks)}" for s in STRATEGIES)
    print(f"Index '{INDEX_NAME}': uploaded {sum(r.succeeded for r in results)} of {len(chunks)} chunks ({counts}), "
          f"{len(vectors[0])} numbers per vector.")


def compare_modes(question: str) -> None:
    """The same question in all three modes, top 3, heading chunks."""
    print(f"\nQuestion: {question}")
    vector = embed([question])[0]                                               # embed once, reuse for vector and hybrid
    for mode in MODES:
        print(f" [{mode}]")
        show_matches(search(question, mode=mode, top=3, vector=vector))


if __name__ == "__main__":
    if "--build" in sys.argv:
        build_index()
        sys.exit()
    if len(sys.argv) > 1:
        for question in sys.argv[1:]:
            compare_modes(question)
        sys.exit()

    print("=== 1. Keyword vs vector vs hybrid ===")
    for question in ["FESTIVE10",
                     "My parcel is stuck and not moving",
                     "Is a fridge compressor covered?"]:
        compare_modes(question)
    print("\nScores are not comparable between modes: BM25 is unbounded, vector scores come from the cosine distance,")
    print("and hybrid (RRF) scores are always small (about 0.03), because they only reflect ranks.")

    print()
    print("=== 2. top-k: how many chunks to send to the model ===")
    question = "I am a Plus member. Do I pay for express delivery, and what would it cost otherwise?"
    print(f"Question: {question}")
    matches = search(question, top=5)
    for k in (1, 3, 5):
        print(f"  top {k}: {[m.source for m in matches[:k]]}")
    print("  Too few chunks: half of a two-part answer is missing. Too many: more tokens, more noise, slower.")

    print()
    print("=== 3. One index, three chunking strategies (a filter) ===")
    question = "How much does ShopEase Protect cost?"
    for strategy in STRATEGIES:
        best = search(question, top=1, chunking=strategy)[0]
        print(f"  chunking eq '{strategy}':  {len(best.content):>4} chars  {best.content[:90]}...")

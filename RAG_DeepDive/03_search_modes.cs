using Azure;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;
using static RAG_DeepDive.AzureClient;
using static RAG_DeepDive.Rag;

namespace RAG_DeepDive;

// Retrieval with Azure AI Search. Keyword vs vector vs hybrid, and top-k.
//
//   keyword   BM25 on the words. Great for codes, names and numbers (FESTIVE10, 1800-200-7327).
//             Misses synonyms: "fridge" does not match "refrigerator".
//   vector    nearest embeddings. Finds the meaning in other words ("parcel stuck" -> "Held at hub").
//             Can miss exact codes and rare words.
//   hybrid    both at once; Azure AI Search merges the two ranked lists with
//             Reciprocal Rank Fusion (RRF): score = sum of 1 / (60 + rank) over both lists.
//
// In code the three modes differ in two places (see SearchAsync in Rag.cs):
//   keyword   the search text: SearchAsync(question, options)
//   vector    options.VectorSearch with a VectorizedQuery, and null as the search text
//   hybrid    both
//
// The shared index "shopease-rag" holds the chunks of ALL three chunking strategies
// from lab 2, with a "chunking" field to filter on. The trainer builds it once.
//
//   dotnet run -- 3                      sample questions
//   dotnet run -- 3 "your question"
//   dotnet run -- 3 --build              trainer only: needs Search:AdminKey (or env var Search__AdminKey)
class SearchModes
{
    public static async Task RunAsync()
    {
        if (Program.Args.Contains("--build"))
        {
            await BuildIndexAsync();
            return;
        }
        if (Program.Args.Length > 0)
        {
            foreach (var question in Program.Args)
                await CompareModesAsync(question);
            return;
        }

        Console.WriteLine("=== 1. Keyword vs vector vs hybrid ===");
        foreach (var question in new[] { "FESTIVE10", "My parcel is stuck and not moving", "Is a fridge compressor covered?" })
            await CompareModesAsync(question);
        Console.WriteLine("\nScores are not comparable between modes: BM25 is unbounded, vector scores come from the cosine distance,");
        Console.WriteLine("and hybrid (RRF) scores are always small (about 0.03), because they only reflect ranks.");

        Console.WriteLine();
        Console.WriteLine("=== 2. top-k: how many chunks to send to the model ===");
        var plusQuestion = "I am a Plus member. Do I pay for express delivery, and what would it cost otherwise?";
        Console.WriteLine($"Question: {plusQuestion}");
        var matches = await SearchAsync(plusQuestion, top: 5);
        foreach (var k in new[] { 1, 3, 5 })
            Console.WriteLine($"  top {k}: [{string.Join(", ", matches.Take(k).Select(m => $"'{m.Source}'"))}]");
        Console.WriteLine("  Too few chunks: half of a two-part answer is missing. Too many: more tokens, more noise, slower.");

        Console.WriteLine();
        Console.WriteLine("=== 3. One index, three chunking strategies (a filter) ===");
        var protectQuestion = "How much does ShopEase Protect cost?";
        foreach (var strategy in Strategies)
        {
            var best = (await SearchAsync(protectQuestion, top: 1, chunking: strategy))[0];
            Console.WriteLine($"  chunking eq '{strategy}':  {best.Content.Length,4} chars  {best.Content[..Math.Min(90, best.Content.Length)]}...");
        }
    }

    // The same question in all three modes, top 3, heading chunks.
    private static async Task CompareModesAsync(string question)
    {
        Console.WriteLine();
        Console.WriteLine($"Question: {question}");
        var vector = (await EmbedAsync([question]))[0];                  // embed once, reuse for vector and hybrid
        foreach (var mode in Modes)
        {
            Console.WriteLine($" [{mode}]");
            ShowMatches(await SearchAsync(question, mode: mode, top: 3, vector: vector));
        }
    }

    // Trainer only: chunk with every strategy, embed, and upload to a fresh index. Safe to run twice.
    private static async Task BuildIndexAsync()
    {
        var endpoint = new Uri(Setting("Search:Endpoint"));
        var credential = new AzureKeyCredential(Setting("Search:AdminKey"));
        var chunks = Strategies.SelectMany(strategy => ChunkAll(strategy)).ToList();
        var vectors = await EmbedAsync(chunks.Select(chunk => chunk.Content));          // one call for all chunks

        var index = new SearchIndex(IndexName)
        {
            Fields =
            {
                new SimpleField("id", SearchFieldDataType.String) { IsKey = true },
                new SimpleField("chunking", SearchFieldDataType.String) { IsFilterable = true },
                new SimpleField("source", SearchFieldDataType.String) { IsFilterable = true },
                new SearchableField("title"),                                           // keyword search
                new SearchableField("content"),                                         // keyword search
                new SearchField("vector", SearchFieldDataType.Collection(SearchFieldDataType.Single))
                {
                    IsSearchable = true,
                    VectorSearchDimensions = vectors[0].Length,
                    VectorSearchProfileName = "default",                               // vector search
                },
            },
            VectorSearch = new VectorSearch
            {
                Algorithms = { new HnswAlgorithmConfiguration("hnsw") },               // cosine similarity by default
                Profiles = { new VectorSearchProfile("default", "hnsw") },
            },
            SemanticSearch = new SemanticSearch                                         // for the semantic ranker (lab 4)
            {
                Configurations =
                {
                    new SemanticConfiguration("default", new SemanticPrioritizedFields
                    {
                        TitleField = new SemanticField("title"),
                        ContentFields = { new SemanticField("content") },
                    }),
                },
            },
        };

        var indexes = new SearchIndexClient(endpoint, credential);
        var existing = new List<string>();
        await foreach (var name in indexes.GetIndexNamesAsync())
            existing.Add(name);
        if (existing.Contains(IndexName))
            await indexes.DeleteIndexAsync(IndexName, CancellationToken.None);         // start clean: old chunk ids go too
        try
        {
            await indexes.CreateIndexAsync(index);
        }
        catch (RequestFailedException error)
        {
            Console.WriteLine($"Could not add the semantic configuration ({error.Message.Split('\n')[0]}). Creating the index without it.");
            index.SemanticSearch = null;
            await indexes.CreateIndexAsync(index);
        }

        var documents = chunks.Select((chunk, i) => new SearchDocument
        {
            ["id"] = chunk.Id,
            ["chunking"] = chunk.Chunking,
            ["source"] = chunk.Source,
            ["title"] = chunk.Title,
            ["content"] = chunk.Content,
            ["vector"] = vectors[i],
        });
        var result = await new SearchClient(endpoint, IndexName, credential).UploadDocumentsAsync(documents);
        var counts = string.Join(", ", Strategies.Select(s => $"{s} {chunks.Count(c => c.Chunking == s)}"));
        Console.WriteLine($"Index '{IndexName}': uploaded {result.Value.Results.Count(r => r.Succeeded)} of {chunks.Count} chunks ({counts}), " +
                          $"{vectors[0].Length} numbers per vector.");
    }
}

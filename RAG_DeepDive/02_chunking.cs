using static RAG_DeepDive.AzureClient;
using static RAG_DeepDive.Rag;

namespace RAG_DeepDive;

// How you cut the documents decides what retrieval can find.
//
// Three common strategies (the code is in Rag.cs, about ten lines each):
//   fixed       every N characters with some overlap; ignores the structure
//   paragraph   one chunk per paragraph
//   heading     one chunk per "## " section, the heading kept in the title
//
// Too small: a chunk loses its context ("It costs 8%" - what does?).
// Too big:   one chunk covers many topics, its embedding gets blurry, and every
//            question sends the model more text than it needs (tokens = money).
//
// This lab searches in memory with cosine similarity, like the Ollama day, so
// you can try any setting without rebuilding an index.
class Chunking
{
    public static async Task RunAsync()
    {
        Console.WriteLine("=== 1. The same seven documents, cut three ways ===");
        foreach (var strategy in Strategies)
        {
            var lengths = ChunkAll(strategy).Select(c => c.Content.Length).ToList();
            Console.WriteLine($"  {strategy,-10} {lengths.Count,3} chunks, average {lengths.Sum() / lengths.Count,4} characters, longest {lengths.Max(),5}");
        }

        Console.WriteLine();
        Console.WriteLine("=== 2. Where the fact about ShopEase Protect's price ends up ===");
        foreach (var strategy in Strategies)
        {
            var chunk = ChunkAll(strategy).First(c => c.Content.Contains("8% of"));
            var body = chunk.Content[(chunk.Title.Length + 2)..];
            var at = body.IndexOf("8% of", StringComparison.Ordinal);
            Console.WriteLine($"  [{strategy}] {chunk.Id}, {chunk.Content.Length} characters, title '{chunk.Title}'");
            Console.WriteLine($"    starts: {body[..Math.Min(50, body.Length)]}...");
            Console.WriteLine($"    fact:   ...{body[Math.Max(0, at - 70)..Math.Min(body.Length, at + 30)]}...");
        }
        Console.WriteLine("  The fixed chunk starts mid-sentence: does it still say WHAT costs 8%? Only the heading chunk's title does.");

        Console.WriteLine();
        Console.WriteLine("=== 3. Where fixed-size cuts fall (300 characters, 50 overlap) ===");
        foreach (var chunk in ChunkAll("fixed").Where(c => c.Source == "warranty-and-repairs.md").Take(3))
        {
            var body = chunk.Content[(chunk.Title.Length + 2)..];
            Console.WriteLine($"  starts '{body[..28]}'  ends '{body[^28..]}'");
        }
        Console.WriteLine("  The overlap repeats the last 50 characters of each chunk at the start of the next one.");

        Console.WriteLine();
        Console.WriteLine("=== 4. Does retrieval get better or worse? ===");
        (string Name, string Strategy, int Size, int Overlap)[] configs =
        [
            ("fixed 100, no overlap", "fixed", 100, 0),
            ("fixed 300, no overlap", "fixed", 300, 0),
            ("fixed 300, overlap 50", "fixed", 300, 50),
            ("fixed 1000, overlap 100", "fixed", 1000, 100),
            ("paragraph", "paragraph", 0, 0),
            ("heading", "heading", 0, 0),
        ];
        var cases = TestSet.Where(c => c.Source is not null).ToList();         // the questions that have an answer
        var questionVectors = await EmbedAsync(cases.Select(c => c.Question)); // embed the questions once
        Console.WriteLine($"{cases.Count} questions, top 3 chunks each. 'File' = right document first; 'Fact' = the fact is in the 3 chunks.");
        Console.WriteLine($"  {"Chunking",-24} {"Chunks",6} {"Avg chars",9} {"File@1",7} {"Fact@3",7}  {"Text sent per question",22}");
        foreach (var (name, strategy, size, overlap) in configs)
        {
            var chunks = strategy == "fixed" ? ChunkAll(strategy, size, overlap) : ChunkAll(strategy);
            var vectors = await EmbedAsync(chunks.Select(c => c.Content));      // one request per configuration
            int fileHits = 0, factHits = 0, sent = 0;
            for (var q = 0; q < cases.Count; q++)
            {
                var top = chunks.Zip(vectors)
                    .OrderByDescending(pair => CosineSimilarity(questionVectors[q], pair.Second))
                    .Take(3).Select(pair => pair.First).ToList();
                fileHits += RankOf(cases[q], top.Select(c => c.Source)) == 1 ? 1 : 0;
                factHits += FactFound(cases[q], top.Select(c => c.Content)) ? 1 : 0;
                sent += top.Sum(c => c.Content.Length);
            }
            var average = chunks.Sum(c => c.Content.Length) / chunks.Count;
            Console.WriteLine($"  {name,-24} {chunks.Count,6} {average,9} {fileHits,4}/{cases.Count} {factHits,4}/{cases.Count}" +
                              $"  {sent / cases.Count,14} chars ≈ {sent / cases.Count / 4} tokens");
        }

        Console.WriteLine();
        Console.WriteLine("Read the table: tiny chunks lose facts at their edges, huge chunks blur the meaning and cost more tokens.");
        Console.WriteLine("Cut where the author changed topic, keep the title in every chunk, and add overlap when you must cut by size.");
    }
}

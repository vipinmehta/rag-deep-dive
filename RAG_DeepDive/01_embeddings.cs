using OpenAI.Embeddings;
using static RAG_DeepDive.AzureClient;
using static RAG_DeepDive.Rag;

namespace RAG_DeepDive;

// An embedding model turns a text into a list of numbers (a vector). Texts with a
// similar meaning get vectors that point in a similar direction, even when they
// share no words. Cosine similarity measures that direction:
//
//   cos(a, b) = (a · b) / (|a| × |b|)          1 = same meaning, 0 = unrelated
//
// On the Ollama day nomic-embed-text gave 768 numbers. The class's "embed"
// deployment (text-embedding-3-small) gives 1,536, and can give fewer if you ask.
class EmbeddingBasics
{
    public static async Task RunAsync()
    {
        Console.WriteLine("=== 1. One sentence becomes 1,536 numbers ===");
        OpenAIEmbeddingCollection reply = await Embeddings.GenerateEmbeddingsAsync(["My parcel is late."]);
        var vector = reply[0].ToFloats().ToArray();
        Console.WriteLine($"Model:      {reply.Model}");
        Console.WriteLine($"Dimensions: {vector.Length}");
        Console.WriteLine($"First 6:    [{string.Join(", ", vector.Take(6).Select(x => Math.Round(x, 4)))}] ...");
        Console.WriteLine($"Length:     {Math.Sqrt(vector.Sum(x => (double)x * x)):F4}   (always 1: the vector is normalised, so only its direction counts)");
        Console.WriteLine($"Tokens:     {reply.Usage.InputTokenCount}   (embeddings bill input tokens only: about USD 0.02 per million)");

        Console.WriteLine();
        Console.WriteLine("=== 2. Similar vs unrelated sentences ===");
        const string Anchor = "My parcel is late.";
        (string Sentence, string Kind)[] others =
        [
            ("My order still has not arrived.", "same meaning, no shared words"),
            ("Where is my delivery?", "same topic, a question"),
            ("मेरा पार्सल देर से आया है।", "same meaning, in Hindi"),
            ("My parcel is lovely.", "shared words, different meaning"),
            ("How do I reset my password?", "ShopEase, but another topic"),
            ("The cat sat on the mat.", "unrelated"),
        ];
        var vectors = await EmbedAsync([Anchor, .. others.Select(o => o.Sentence)]);   // one request for all seven
        Console.WriteLine($"Compared with: '{Anchor}'");
        for (var i = 0; i < others.Length; i++)
        {
            var score = CosineSimilarity(vectors[0], vectors[i + 1]);
            Console.WriteLine($"  {score:F3}  {new string('#', (int)Math.Round(score * 30)),-30}  {others[i].Sentence,-34} ({others[i].Kind})");
        }
        Console.WriteLine("Read the gaps, not the absolute values: with this model even 'unrelated' rarely drops below 0.05.");

        Console.WriteLine();
        Console.WriteLine("=== 3. Fewer dimensions: smaller vectors, almost the same ranking ===");
        var shortVectors = await EmbedAsync([Anchor, .. others.Select(o => o.Sentence)], dimensions: 256);
        Console.WriteLine($"{"1,536 dims",10}  {"256 dims",8}   sentence");
        for (var i = 0; i < others.Length; i++)
            Console.WriteLine($"{CosineSimilarity(vectors[0], vectors[i + 1]),10:F3}  {CosineSimilarity(shortVectors[0], shortVectors[i + 1]),8:F3}   {others[i].Sentence}");
        Console.WriteLine($"Storage per chunk: 1,536 × 4 bytes = {1536 * 4:N0} bytes, 256 × 4 bytes = {256 * 4:N0} bytes.");
        Console.WriteLine("The index must use the SAME model and the SAME dimensions for the chunks and for every question.");

        Console.WriteLine();
        Console.WriteLine("=== 4. Cosine similarity is just a dot product here ===");
        var dot = vectors[0].Zip(vectors[1], (x, y) => (double)x * y).Sum();
        Console.WriteLine($"dot product {dot:F6}  =  cosine {CosineSimilarity(vectors[0], vectors[1]):F6}   (both vectors have length 1)");
    }
}

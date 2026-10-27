using System.Text.Json;
using Azure;
using OpenAI.Chat;
using static RAG_DeepDive.AzureClient;
using static RAG_DeepDive.Rag;

namespace RAG_DeepDive;

// Reranking. Retrieve many, reorder carefully, keep a few.
//
// Search is fast because it compares vectors and words, not meaning in depth.
// A reranker reads the question and each candidate TOGETHER and scores how well
// the candidate answers it. Slower and costlier, so only for the top 10-50.
//
//   Part 1  an LLM reranker: the chat model scores 10 hybrid candidates 0-3.
//           Works on any tier, costs one chat call per question.
//   Part 2  the Azure AI Search semantic ranker: the same idea as a built-in feature
//           (reranker score 0-4). It needs the Basic tier or higher; on the class's
//           Free tier this part prints why it is not available.
//
//   dotnet run -- 4                      sample questions
//   dotnet run -- 4 "your question"
class Reranking
{
    private const int Candidates = 10;
    private const int Keep = 3;

    private static readonly object RerankSchema = new
    {
        type = "object",
        properties = new
        {
            scores = new
            {
                type = "array",
                items = new
                {
                    type = "object",
                    properties = new { source = new { type = "integer" }, relevance = new { type = "integer", @enum = new[] { 0, 1, 2, 3 } } },
                    required = new[] { "source", "relevance" },
                    additionalProperties = false,
                },
            },
        },
        required = new[] { "scores" },
        additionalProperties = false,
    };

    public static async Task RunAsync()
    {
        string[] questions = Program.Args.Length > 0
            ? Program.Args
            : ["Is my phone covered if I drop it and the screen cracks?", "What is the capital of France?"];
        foreach (var question in questions)
            await DemoAsync(question);
        Console.WriteLine("\nSearch ALWAYS returns something, even for France. A reranker score (LLM 0-3, semantic 0-4)");
        Console.WriteLine("says how relevant the best chunk really is: a good signal for 'I don't know' (lab 5).");
    }

    // Ask the chat model for a relevance score per candidate, then sort by it (ties keep the search order).
    private static async Task<List<(int Relevance, Match Match)>> LlmRerankAsync(string question, List<Match> matches)
    {
        var numbered = string.Join("\n\n", matches.Select((m, i) => $"[{i + 1}] {m.Content}"));
        var reply = await ChatAsync(
            [
                new SystemChatMessage("Score how well each numbered text answers the question. 3 = answers it, " +
                                      "2 = partly, 1 = same topic only, 0 = unrelated. Score every text."),
                new UserChatMessage($"Question: {question}\n\nTexts:\n{numbered}"),
            ],
            JsonSchema("rerank", RerankSchema));
        var relevance = new Dictionary<int, int>();
        foreach (var item in JsonDocument.Parse(reply).RootElement.GetProperty("scores").EnumerateArray())
            relevance[item.GetProperty("source").GetInt32()] = item.GetProperty("relevance").GetInt32();
        return matches.Select((m, i) => (relevance.GetValueOrDefault(i + 1), m))
            .OrderByDescending(pair => pair.Item1)                                      // OrderBy is stable
            .ToList();
    }

    private static async Task DemoAsync(string question)
    {
        Console.WriteLine();
        Console.WriteLine($"Question: {question}");
        var matches = await SearchAsync(question, top: Candidates);
        Console.WriteLine($" Hybrid search, top {Keep} of {Candidates}:");
        ShowMatches(matches.Take(Keep));

        Console.WriteLine($" After the LLM reranker (relevance 0-3), top {Keep}:");
        foreach (var (relevance, m) in (await LlmRerankAsync(question, matches)).Take(Keep))
            Console.WriteLine($"  relevance {relevance}  (was rank {matches.IndexOf(m) + 1,2})  {m.Title}");

        Console.WriteLine(" After the Azure semantic ranker:");
        try
        {
            ShowMatches(await SearchAsync(question, top: Keep, semantic: true));
        }
        catch (RequestFailedException error)
        {
            Console.WriteLine($"  Not available: {SearchErrorHint(error)}");
        }
    }
}

using System.Text.Json;
using OpenAI.Chat;
using static RAG_DeepDive.AzureClient;
using static RAG_DeepDive.Rag;

namespace RAG_DeepDive;

// Evaluate retrieval and answers on a small test set.
//
// "It answered my three questions" is not a test. Fourteen questions with known
// answers (TestSet in Rag.cs) tell you whether a change made things better or worse.
//
//   retrieval   hit rate@3   the right file is in the top 3
//               fact@3       the fact itself is in the 3 chunks (a badly cut chunk can miss it)
//               MRR          mean of 1/rank of the right file: 1.0 = always first
//   answers     correct      contains the expected fact, or says NoAnswer when it should
//   grounding   an LLM judge checks every claim in the answer against the sources
//               (structured output, so code can count the verdicts)
//
//   dotnet run -- 6
//   dotnet run -- 6 --semantic           with the semantic ranker: Basic tier or higher
class Evaluate
{
    private const int TopK = 3;

    private static readonly object JudgeSchema = new
    {
        type = "object",
        properties = new
        {
            grounded = new { type = "boolean" },
            unsupported_claims = new { type = "array", items = new { type = "string" } },
        },
        required = new[] { "grounded", "unsupported_claims" },
        additionalProperties = false,
    };

    private record Verdict(bool Grounded, List<string> UnsupportedClaims);

    // LLM-as-judge: is every claim in the answer stated in the sources?
    private static async Task<Verdict> JudgeAsync(string answer, List<Match> matches)
    {
        var reply = await ChatAsync(
            [
                new SystemChatMessage("You check a support answer against its sources. grounded = true only if every " +
                                      "factual claim in the answer is stated in the sources. Citation numbers like [1] " +
                                      $"are not claims. The reply '{NoAnswer}' is always grounded. " +
                                      "List each unsupported claim in a few words."),
                new UserChatMessage($"Sources:\n{FormatSources(matches)}\n\nAnswer:\n{answer}"),
            ],
            JsonSchema("verdict", JudgeSchema));
        return JsonSerializer.Deserialize<Verdict>(reply, new JsonSerializerOptions { PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower })!;
    }

    public static async Task RunAsync()
    {
        var semantic = Program.Args.Contains("--semantic");
        int hits = 0, facts = 0, correct = 0, grounded = 0;
        var reciprocalRanks = new List<double>();
        Console.WriteLine($"Hybrid search{(semantic ? " + semantic ranker" : "")}, heading chunks, top {TopK}\n");

        foreach (var test in TestSet)
        {
            var matches = await SearchAsync(test.Question, top: TopK, semantic: semantic);
            var best = matches.Select(m => m.Similarity ?? 0).DefaultIfEmpty(0).Max();
            var answer = best < MinSimilarity ? NoAnswer : await GenerateAsync(test.Question, matches);

            bool ok;
            string retrieved;
            if (test.Source is not null)                                 // retrieval: only for answerable questions
            {
                var rank = RankOf(test, matches.Select(m => m.Source));
                hits += rank > 0 ? 1 : 0;
                facts += FactFound(test, matches.Select(m => m.Content)) ? 1 : 0;
                reciprocalRanks.Add(rank > 0 ? 1.0 / rank : 0);
                ok = test.Facts.Any(fact => answer.Contains(fact, StringComparison.OrdinalIgnoreCase));
                retrieved = rank > 0 ? $"rank {rank}" : "MISSED";
            }
            else
            {
                ok = answer.Contains(NoAnswer, StringComparison.OrdinalIgnoreCase);
                retrieved = "no answer";
            }
            var verdict = answer != NoAnswer ? await JudgeAsync(answer, matches) : new Verdict(true, []);
            correct += ok ? 1 : 0;
            grounded += verdict.Grounded ? 1 : 0;

            Console.WriteLine($"  {(ok ? "PASS" : "FAIL")}  {retrieved,-9} {(verdict.Grounded ? "grounded" : "NOT GROUNDED"),-12} {test.Question}");
            if (!ok)
                Console.WriteLine($"        expected [{string.Join(", ", test.Facts.DefaultIfEmpty(NoAnswer))}], got: {answer[..Math.Min(100, answer.Length)]}");
            foreach (var claim in verdict.UnsupportedClaims)
                Console.WriteLine($"        unsupported: {claim}");
        }

        var answerable = TestSet.Count(t => t.Source is not null);
        Console.WriteLine();
        Console.WriteLine($"  Retrieval   hit rate@{TopK} {hits}/{answerable}   fact@{TopK} {facts}/{answerable}   MRR {reciprocalRanks.Sum() / answerable:F2}");
        Console.WriteLine($"  Answers     correct {correct}/{TestSet.Length}");
        Console.WriteLine($"  Grounding   judged grounded {grounded}/{TestSet.Length}");
        Console.WriteLine("\nWrite these numbers down. Change ONE thing (top_k, chunking, prompt, gate), run again, compare.");
        Console.WriteLine("A judge is a model too: read the FAILs and the unsupported claims yourself before you trust the totals.");
    }
}

using Azure;
using static RAG_DeepDive.AzureClient;
using static RAG_DeepDive.Rag;

namespace RAG_DeepDive;

// Tune retrieval with numbers. Chunking × search mode × top-k.
//
// Lab 6 measured one configuration end to end. Most RAG problems are retrieval
// problems, and retrieval is cheap to measure: no chat model, one search per
// question. This lab runs the answerable questions of the test set against
// every chunking strategy in every search mode, and prints one table.
//
// One search with top 5 gives hit@1, hit@3 and hit@5 at once: the first 1, 3 or 5 results.
//
//   dotnet run -- 7
//   dotnet run -- 7 --semantic           adds hybrid + semantic ranker: Basic tier or higher
class Tune
{
    public static async Task RunAsync()
    {
        var cases = TestSet.Where(c => c.Source is not null).ToList();
        var vectors = await EmbedAsync(cases.Select(c => c.Question));             // embed the questions once, reuse everywhere
        var configs = Strategies.SelectMany(strategy => Modes.Select(mode => (Strategy: strategy, Mode: mode, Semantic: false))).ToList();
        if (Program.Args.Contains("--semantic"))
            configs.AddRange(Strategies.Select(strategy => (strategy, "hybrid", true)));

        Console.WriteLine($"{cases.Count} questions, {configs.Count} configurations, {cases.Count * configs.Count} searches, no chat calls.\n");
        Console.WriteLine($"  {"Chunking",-10} {"Mode",-16} {"hit@1",6} {"hit@3",6} {"hit@5",6} {"fact@3",7} {"MRR",5}");
        var results = new List<(double Mrr, string Strategy, string Mode)>();
        foreach (var (strategy, mode, semantic) in configs)
        {
            var ranks = new List<int>();
            var factHits = 0;
            try
            {
                for (var q = 0; q < cases.Count; q++)
                {
                    var matches = await SearchAsync(cases[q].Question, mode: mode, top: 5, chunking: strategy, semantic: semantic, vector: vectors[q]);
                    ranks.Add(RankOf(cases[q], matches.Select(m => m.Source)));
                    factHits += FactFound(cases[q], matches.Take(3).Select(m => m.Content)) ? 1 : 0;
                }
            }
            catch (RequestFailedException error)
            {
                Console.WriteLine($"  {strategy,-10} {"hybrid+semantic",-16} not available: {SearchErrorHint(error)}");
                continue;
            }
            int Hit(int k) => ranks.Count(r => r > 0 && r <= k);
            var mrr = ranks.Where(r => r > 0).Sum(r => 1.0 / r) / cases.Count;
            var name = mode + (semantic ? "+semantic" : "");
            results.Add((mrr, strategy, name));
            Console.WriteLine($"  {strategy,-10} {name,-16} {Hit(1),3}/{cases.Count} {Hit(3),3}/{cases.Count} {Hit(5),3}/{cases.Count}" +
                              $" {factHits,4}/{cases.Count} {mrr,5:F2}");
        }

        if (results.Count > 0)
        {
            var best = results.MaxBy(r => r.Mrr);
            Console.WriteLine($"\nBest on this test set: {best.Strategy} chunks, {best.Mode} search (MRR {best.Mrr:F2}).");
        }
        Console.WriteLine("Eleven questions is a tiny test set: one question is 9 points. Before you trust a winner, add questions");
        Console.WriteLine("from real users, especially the ones that FAILED, and run the table again.");
    }
}

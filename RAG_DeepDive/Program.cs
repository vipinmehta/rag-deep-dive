using Azure;

namespace RAG_DeepDive;

// Lab menu. Run one lab directly:   dotnet run -- 3
// Or run without a number to pick from the list (works with F5 in Visual Studio too).
// Lab 3 has a trainer-only extra:      dotnet run -- 3 --build
class Program
{
    private static readonly (string Name, Func<Task> Run)[] Labs =
    [
        ("Embeddings: meaning as numbers", EmbeddingBasics.RunAsync),
        ("Chunking: fixed, paragraph, heading", Chunking.RunAsync),
        ("Search modes: keyword, vector, hybrid, top-k", SearchModes.RunAsync),
        ("Reranking: LLM reranker and semantic ranker", Reranking.RunAsync),
        ("Grounded answers, citations, \"I don't know\"", GroundedAnswers.RunAsync),
        ("Evaluate: hit rate, MRR, groundedness", Evaluate.RunAsync),
        ("Tune: chunking × mode × top-k", Tune.RunAsync),
    ];

    public static string[] Args { get; private set; } = [];

    static async Task Main(string[] args)
    {
        Console.OutputEncoding = System.Text.Encoding.UTF8;      // lab 1 prints Hindi text
        Args = args.Skip(1).ToArray();                           // anything after the lab number

        if (args.Length > 0)
        {
            await RunLabAsync(args[0]);
            return;
        }

        while (true)
        {
            Console.WriteLine();
            for (var i = 0; i < Labs.Length; i++)
                Console.WriteLine($"  {i + 1}. {Labs[i].Name}");
            Console.Write("Lab number (Enter to quit): ");

            var choice = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(choice))
                return;
            await RunLabAsync(choice);
        }
    }

    private static async Task RunLabAsync(string choice)
    {
        if (!int.TryParse(choice, out var number) || number < 1 || number > Labs.Length)
        {
            Console.WriteLine($"Choose a lab from 1 to {Labs.Length}.");
            return;
        }

        var (name, run) = Labs[number - 1];
        Console.WriteLine($"=== Lab {number}: {name} ===");
        try
        {
            await run();
        }
        catch (InvalidOperationException error) when (error.Message.StartsWith("Setting ") || error.Message.StartsWith("Index "))
        {
            Console.WriteLine(error.Message);                    // a missing key or endpoint, or the index is not built
        }
        catch (RequestFailedException error)
        {
            Console.WriteLine($"Azure AI Search error {error.Status}: {error.Message.Split('\n')[0]}");
        }
    }
}

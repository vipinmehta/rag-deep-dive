using OpenAI.Chat;
using static RAG_DeepDive.AzureClient;
using static RAG_DeepDive.Rag;

namespace RAG_DeepDive;

// Grounding, citations and "I don't know".
//
// Retrieval found the right chunks. Now the prompt must make the model use them,
// and only them:
//
//   grounding   "Answer using ONLY the numbered sources below" + the sources in the message
//   citations   numbered sources, the model writes [1] [2]; your CODE checks every number
//   no answer   two layers:
//                 1. a retrieval gate: if no chunk is similar enough, do not call the model at all
//                 2. the prompt: if the sources do not say it, reply exactly NoAnswer
//
//   dotnet run -- 5                      sample questions
//   dotnet run -- 5 "your question"
class GroundedAnswers
{
    public static async Task RunAsync()
    {
        if (Program.Args.Length > 0)
        {
            foreach (var question in Program.Args)
                await AnswerWithCitationsAsync(question);
            return;
        }

        Console.WriteLine("=== 1. Without and with grounding ===");
        var lateQuestion = "My order arrived 5 days late. What compensation do I get?";
        Console.WriteLine($"Q: {lateQuestion}");
        Console.WriteLine($"No sources:   {await ChatAsync([new UserChatMessage($"I shop at ShopEase. {lateQuestion} Answer in two sentences.")])}");
        Console.WriteLine($"With sources: {await GenerateAsync(lateQuestion, await SearchAsync(lateQuestion, top: 3))}");
        Console.WriteLine("Without sources the model gives a plausible, generic answer. Only the ShopEase documents know 'Rs. 150 voucher'.");

        Console.WriteLine();
        Console.WriteLine("=== 2. Citations you can check ===");
        await AnswerWithCitationsAsync("I am a Plus member. Do I pay for express delivery, and what would it cost otherwise?");
        await AnswerWithCitationsAsync("How long is the warranty on a fridge compressor?");

        Console.WriteLine();
        Console.WriteLine("=== 3. When the answer is not in the documents ===");
        await AnswerWithCitationsAsync("What is the capital of France?");               // off topic: the gate should stop it
        await AnswerWithCitationsAsync("Who is the CEO of ShopEase?");                  // on topic, not in the documents
        await AnswerWithCitationsAsync("Can I pay with PayPal?");                       // the documents list what you CAN use
        Console.WriteLine("\nDiscuss the PayPal answer: is 'PayPal is not in the list of payment methods' an answer or an 'I don't know'?");
        Console.WriteLine("Decide what you want, then write it into the prompt and into your test set (lab 6).");
    }

    // The full pipeline: retrieve, gate, generate, check the citations.
    private static async Task<string> AnswerWithCitationsAsync(string question)
    {
        var matches = await SearchAsync(question, top: 3);
        var best = matches.Select(m => m.Similarity ?? 0).DefaultIfEmpty(0).Max();
        Console.WriteLine();
        Console.WriteLine($"Q: {question}");
        Console.WriteLine($"   best similarity {best:F3} (gate {MinSimilarity})");
        if (best < MinSimilarity)
        {
            Console.WriteLine($"A: {NoAnswer}   <- layer 1: the gate. No chunk is about this; the model was not called.");
            return NoAnswer;
        }

        var answer = await GenerateAsync(question, matches);
        Console.WriteLine($"A: {answer}");
        if (answer.Trim() == NoAnswer)
        {
            Console.WriteLine("   <- layer 2: the model found nothing in the sources and said so.");
            return answer;
        }

        var cited = CitedNumbers(answer);
        foreach (var n in cited)
        {
            if (n >= 1 && n <= matches.Count)
                Console.WriteLine($"   [{n}] {matches[n - 1].Source} | {matches[n - 1].Title}");
            else
                Console.WriteLine($"   [{n}] DOES NOT EXIST: only {matches.Count} sources were sent. Treat the answer as suspect.");
        }
        if (cited.Count == 0)
            Console.WriteLine("   No citations: the prompt asked for them. Show it with a warning, or ask again.");
        return answer;
    }
}

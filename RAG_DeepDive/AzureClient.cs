using System.ClientModel;
using Microsoft.Extensions.Configuration;
using OpenAI;
using OpenAI.Chat;
using OpenAI.Embeddings;

namespace RAG_DeepDive;

// Shared code for every lab: which Azure resource to call, which deployments
// to use, and small helpers that send a chat or an embeddings request.
//
// The same helpers as AzureClient.cs in AzureAI_Foundry (ChatAsync, ChatRawAsync,
// EmbedAsync), calling the class's Azure OpenAI deployments through the official
// OpenAI package and the v1 API: {endpoint}/openai/v1/
//
// Settings are read in this order (later ones win):
//   appsettings.json         committed, no secrets
//   appsettings.Local.json   your keys; the SAME file as in AzureAI_Foundry. Never commit it
//   environment variables    AzureOpenAI__ApiKey=... (two underscores replace the colon)
public static class AzureClient
{
    private static readonly IConfiguration Configuration = new ConfigurationBuilder()
        .SetBasePath(AppContext.BaseDirectory)
        .AddJsonFile("appsettings.json", optional: false, reloadOnChange: false)
        .AddJsonFile("appsettings.Local.json", optional: true, reloadOnChange: false)
        .AddEnvironmentVariables()
        .Build();

    // Read one setting and stop with a clear message if it is missing.
    public static string Setting(string key) =>
        Configuration[key] is { Length: > 0 } value && !value.StartsWith('<')
            ? value
            : throw new InvalidOperationException(
                $"Setting '{key}' is missing. Copy appsettings.Local.example.json to appsettings.Local.json " +
                $"and fill in the values from your trainer, or set the environment variable {key.Replace(":", "__")}.");

    public static string Endpoint => Setting("AzureOpenAI:Endpoint").TrimEnd('/');   // https://<resource>.openai.azure.com
    public static string ApiKey => Setting("AzureOpenAI:ApiKey");
    public static string ChatDeployment => Setting("AzureOpenAI:ChatDeployment");   // the deployment name, not the model name
    public static string EmbedDeployment => Setting("AzureOpenAI:EmbedDeployment");

    // gpt-5.x models think before they answer. "none" switches that off: faster and cheaper,
    // and plenty for answering from sources. Leave it empty for a model that does not take
    // this setting (for example gpt-4.1-mini).
    public static string? ReasoningEffort => Configuration["AzureOpenAI:ReasoningEffort"] is { Length: > 0 } value ? value : null;

    // The v1 API: the standard OpenAI client, pointed at your Azure resource.
    // No api-version to pick; the "model" in every request is your DEPLOYMENT name.
    public static Uri BaseUrl => new($"{Endpoint}/openai/v1/");
    public static OpenAIClientOptions ClientOptions => new() { Endpoint = BaseUrl };

    // Lazy: created on first use, so a lab that does not need them never reads their settings.
    private static readonly Lazy<ChatClient> LazyChat = new(() => new ChatClient(ChatDeployment, new ApiKeyCredential(ApiKey), ClientOptions));
    private static readonly Lazy<EmbeddingClient> LazyEmbeddings = new(() => new EmbeddingClient(EmbedDeployment, new ApiKeyCredential(ApiKey), ClientOptions));

    public static ChatClient Chat => LazyChat.Value;
    public static EmbeddingClient Embeddings => LazyEmbeddings.Value;

    // Send a list of messages and return the full reply (with token counts).
    public static async Task<ClientResult<ChatCompletion>> ChatRawAsync(IEnumerable<ChatMessage> messages, ChatCompletionOptions? options = null) =>
        await Chat.CompleteChatAsync(messages, WithDefaults(options));

    // Send a list of messages and return only the model's answer text.
    public static async Task<string> ChatAsync(IEnumerable<ChatMessage> messages, ChatCompletionOptions? options = null)
    {
        ChatCompletion reply = await ChatRawAsync(messages, options);
        return string.Concat(reply.Content.Select(part => part.Text));
    }

    // Turn a list of texts into a list of embeddings, one per text, in ONE request.
    //   dimensions: text-embedding-3 models can return shorter vectors (lab 1). null = the full 1,536.
    public static async Task<List<float[]>> EmbedAsync(IEnumerable<string> texts, int? dimensions = null)
    {
        var options = new EmbeddingGenerationOptions { Dimensions = dimensions };
        OpenAIEmbeddingCollection reply = await Embeddings.GenerateEmbeddingsAsync(texts, options);
        return reply.Select(item => item.ToFloats().ToArray()).ToList();
    }

    private static ChatCompletionOptions WithDefaults(ChatCompletionOptions? options)
    {
        options ??= new ChatCompletionOptions();
        if (ReasoningEffort is not null && options.ReasoningEffortLevel is null)
            options.ReasoningEffortLevel = new ChatReasoningEffortLevel(ReasoningEffort);
        return options;
    }

    // A strict JSON schema for structured output, written as an anonymous object (labs 4 and 6).
    public static ChatCompletionOptions JsonSchema(string name, object schema) => new()
    {
        ResponseFormat = ChatResponseFormat.CreateJsonSchemaFormat(name, BinaryData.FromObjectAsJson(schema), jsonSchemaIsStrict: true),
    };
}

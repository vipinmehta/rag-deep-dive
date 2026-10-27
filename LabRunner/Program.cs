// A web page to run the labs: pick a lab, optionally type a question, press Run, read the output.
//
// The labs themselves do not change: this server simply runs
//   dotnet run --project ..\RAG_DeepDive -- N "question"
// and sends what it prints to the page while it runs.
//
// Run:  dotnet run          then open http://localhost:5021
using System.Diagnostics;
using System.Text;
using System.Text.RegularExpressions;

const string Title = "RAG Deep Dive (Azure) · C# labs";
string[] nextSession = ["6", "7"];      // taught in the next session (RAG eval): they still run, the page marks them

var app = WebApplication.Create(args);
var projectFolder = Path.GetFullPath(Path.Combine(app.Environment.ContentRootPath, "..", "RAG_DeepDive"));   // the labs project
app.UseDefaultFiles();          // "/" shows wwwroot/index.html
app.UseStaticFiles();

app.MapGet("/api/info", () => new { title = Title, labs = ReadLabs() });

// Run one lab and stream its output, line by line, to the page.
app.MapPost("/api/run", async (RunRequest run, HttpResponse response) =>
{
    var start = new ProcessStartInfo("dotnet")
    {
        RedirectStandardOutput = true,
        RedirectStandardError = true,
        StandardOutputEncoding = Encoding.UTF8,
        WorkingDirectory = projectFolder,
    };
    start.ArgumentList.Add("run");
    start.ArgumentList.Add("--");
    start.ArgumentList.Add(run.Id);                  // the lab number
    if (!string.IsNullOrWhiteSpace(run.Question))
        start.ArgumentList.Add(run.Question);

    response.ContentType = "text/plain; charset=utf-8";
    using var process = Process.Start(start)!;
    while (await process.StandardOutput.ReadLineAsync() is { } line)
    {
        await response.WriteAsync(line + "\n");
        await response.Body.FlushAsync();            // send it now, not when the lab ends
    }
    await response.WriteAsync(await process.StandardError.ReadToEndAsync());
    await process.WaitForExitAsync();
    if (process.ExitCode != 0)
        await response.WriteAsync($"\n(The lab stopped with exit code {process.ExitCode}.)\n");
});

Console.WriteLine("Open http://localhost:5021");
app.Run("http://localhost:5021");


// The labs: titles from the menu in Program.cs, descriptions from the comments at the top of each 0N_*.cs file.
List<Lab> ReadLabs()
{
    var titles = Regex.Matches(File.ReadAllText(Path.Combine(projectFolder, "Program.cs")), @"\(""(.+?)"", \w+\.RunAsync\)")
        .Select(m => m.Groups[1].Value.Replace("\\\"", "\""))
        .ToList();
    var files = Directory.GetFiles(projectFolder, "0*.cs").Order().ToList();
    var labs = new List<Lab>();
    for (var i = 0; i < files.Count; i++)
    {
        var comments = File.ReadLines(files[i])
            .SkipWhile(line => !line.StartsWith("//"))                     // skip the using and namespace lines
            .TakeWhile(line => line.StartsWith("//"))
            .Select(line => line.Length > 3 ? line[3..] : "")              // remove "// "
            .ToList();
        var about = string.Join("\n", comments.TakeWhile(line => !line.Contains("dotnet run"))).Trim();
        labs.Add(new Lab(
            Id: (i + 1).ToString(),
            Title: titles[i],
            Description: about,
            TakesQuestion: comments.Any(line => line.Contains("\"your question\"")),
            NextSession: nextSession.Contains((i + 1).ToString())));
    }
    return labs;
}

record Lab(string Id, string Title, string Description, bool TakesQuestion, bool NextSession);   // sent as {"id", "title", ...}
record RunRequest(string Id, string? Question);

html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>yt-fts: Your Ultimate YouTube Search Tool</title>
    <meta name="description" content="Effortlessly search YouTube transcripts with yt-fts, a powerful command-line tool. Find specific keywords, phrases, and even use semantic search for deeper insights.">
    <meta name="keywords" content="YouTube search, transcript search, command-line tool, semantic search, yt-fts, OpenAI, Gemini, ChromaDB, full text search, video summaries">
</head>
<body>

    <h1>yt-fts: Supercharge Your YouTube Research with Advanced Search</h1>

    <p>Unlock the hidden knowledge within YouTube videos with <code>yt-fts</code>, a command-line tool that lets you search and analyze video transcripts. <a href="https://github.com/NotJoeMartinez/yt-fts">Explore the yt-fts repository on GitHub!</a></p>

    <img src="https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14" alt="yt-fts Screenshot" width="500">

    <h2>Key Features</h2>
    <ul>
        <li><b>Full Text Search:</b> Quickly find videos containing specific keywords or phrases within their transcripts.</li>
        <li><b>Semantic Search:</b> Leverage the power of OpenAI or Gemini embeddings for more nuanced and relevant search results.</li>
        <li><b>LLM-Powered Chat Bot:</b> Engage in interactive conversations with a chat bot that uses the semantic search results as context.</li>
        <li><b>Video Summarization:</b> Generate concise summaries of YouTube videos, complete with timestamps and links.</li>
        <li><b>Flexible Data Management:</b> Download, update, and export transcripts for easy organization and analysis.</li>
        <li><b>Advanced Search Syntax:</b> Utilize SQLite's Enhanced Query Syntax for powerful and flexible search queries (e.g., AND, OR, wildcards).</li>
    </ul>

    <h2>Installation</h2>

    <p>Install <code>yt-fts</code> using pip:</p>

    ```bash
    pip install yt-fts
    ```

    <h2>Command Reference</h2>

    <h3><code>download</code> - Download Subtitles</h3>
    <p>Downloads subtitles for a YouTube channel or playlist.</p>
    <ul>
        <li><code>yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"</code> - Download a channel</li>
        <li><code>yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"</code> - Download using cookies.</li>
        <li><code>yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"</code> - Download a playlist</li>
        <li><b>Options:</b> <code>--playlist</code>, <code>--language</code>, <code>--jobs</code>, <code>--cookies-from-browser</code></li>
    </ul>

    <h3><code>diagnose</code> - Diagnose Download Issues</h3>
    <p>Tests your connection to YouTube and provides troubleshooting recommendations.</p>
    <ul>
        <li><code>yt-fts diagnose</code></li>
        <li><code>yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox</code></li>
        <li><b>Options:</b> <code>--test-url</code>, <code>--cookies-from-browser</code>, <code>--jobs</code></li>
    </ul>

    <h3><code>list</code> - List Library Contents</h3>
    <p>Lists saved channels, videos, and transcripts.</p>
    <ul>
        <li><code>yt-fts list</code> - List all channels</li>
        <li><code>yt-fts list --channel "3Blue1Brown"</code> - List videos for a specific channel</li>
        <li><code>yt-fts list --transcript "dQw4w9WgXcQ"</code> - Show transcript for a specific video</li>
        <li><b>Options:</b> <code>--transcript</code>, <code>--channel</code>, <code>--library</code></li>
    </ul>

    <h3><code>update</code> - Update Subtitles</h3>
    <p>Updates subtitles for all or a specific channel.</p>
    <ul>
        <li><code>yt-fts update</code> - Update all channels</li>
        <li><code>yt-fts update --channel "3Blue1Brown" --jobs 5</code> - Update a specific channel</li>
        <li><b>Options:</b> <code>--channel</code>, <code>--language</code>, <code>--jobs</code>, <code>--cookies-from-browser</code></li>
    </ul>

    <h3><code>delete</code> - Delete a Channel</h3>
    <p>Deletes a channel and all its data. Requires confirmation.</p>
    <ul>
        <li><code>yt-fts delete --channel "3Blue1Brown"</code></li>
        <li><b>Options:</b> <code>--channel</code> (required)</li>
    </ul>

    <h3><code>export</code> - Export Transcripts</h3>
    <p>Exports transcripts for a channel in various formats (txt, vtt).</p>
    <ul>
        <li><code>yt-fts export --channel "3Blue1Brown" --format txt</code></li>
        <li><code>yt-fts export --channel "3Blue1Brown" --format vtt</code></li>
        <li><b>Options:</b> <code>--channel</code> (required), <code>--format</code></li>
    </ul>

    <h3><code>search</code> - Full Text Search</h3>
    <p>Performs full text search across saved transcripts using SQLite's enhanced query syntax.</p>
    <ul>
        <li><code>yt-fts search "[search query]"</code> - Search in all channels</li>
        <li><code>yt-fts search "[search query]" --channel "[channel name or id]"</code> - Search in a specific channel</li>
        <li><code>yt-fts search "[search query]" --video-id "[video id]"</code> - Search in a specific video</li>
        <li><code>yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"</code> - Limit Results</li>
        <li><code>yt-fts search "[search query]" --export --channel "[channel name or id]"</code> - Export to CSV</li>
        <li><b>Options:</b> <code>--channel</code>, <code>--video-id</code>, <code>--limit</code>, <code>--export</code></li>
        <li><b>Advanced Search:</b> Use SQLite's <a href="https://www.sqlite.org/fts3.html#full_text_index_queries">Enhanced Query Syntax</a> (AND, OR, wildcards, prefix queries).</li>
    </ul>

    <h3><code>embeddings</code> - Semantic Search Setup</h3>
    <p>Generates embeddings for semantic search, using OpenAI or Gemini. Requires an API key (set via environment variable or `--api-key`).</p>
    <ul>
        <li><code>yt-fts embeddings --channel "3Blue1Brown"</code></li>
        <li><code>yt-fts embeddings --interval 60 --channel "3Blue1Brown"</code></li>
        <li><b>Options:</b> <code>--channel</code>, <code>--api-key</code>, <code>--interval</code></li>
    </ul>

    <h3><code>vsearch</code> - Semantic (Vector) Search</h3>
    <p>Performs semantic search, returning results sorted by similarity. Requires embeddings to be enabled for the channel.</p>
    <ul>
        <li><code>yt-fts vsearch "[search query]" --channel "[channel name or id]"</code></li>
        <li><code>yt-fts vsearch "[search query]" --video-id "[video id]"</code></li>
        <li><code>yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"</code></li>
        <li><code>yt-fts vsearch "[search query]" --export --channel "[channel name or id]"</code></li>
        <li><b>Options:</b> <code>--channel</code>, <code>--video-id</code>, <code>--limit</code>, <code>--export</code>, <code>--api-key</code></li>
    </ul>

    <h3><code>llm</code> - Chat Bot</h3>
    <p>Starts an interactive chat session using the semantic search results of your initial prompt as the context to answer questions.</p>
    <ul>
        <li><code>yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"</code></li>
        <li><b>Options:</b> <code>--channel</code> (required), <code>--api-key</code></li>
    </ul>

    <h3><code>summarize</code> - Summarize Videos</h3>
    <p>Summarizes a YouTube video transcript, providing time stamped URLs.</p>
    <ul>
        <li><code>yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"</code></li>
        <li><code>yt-fts summarize "9-Jl0dxWQs8"</code></li>
        <li><code>yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"</code></li>
        <li><b>Options:</b> <code>--model</code>, <code>--api-key</code></li>
    </ul>

    <h3><code>config</code> - Show Configuration</h3>
    <p>Displays the current configuration settings, including database and ChromaDB paths.</p>
    <ul>
        <li><code>yt-fts config</code></li>
    </ul>

    <h2>How To</h2>

    <h3>Export Search Results</h3>
    <p>Export search results to a CSV file using the <code>--export</code> flag with both <code>search</code> and <code>vsearch</code> commands.</p>
    <ul>
        <li><code>yt-fts search "life in the big city" --export</code></li>
        <li><code>yt-fts vsearch "existing in large metropolaten center" --export</code></li>
    </ul>

    <h3>Delete a Channel</h3>
    <p>Delete a channel with all its associated data.</p>
    <ul>
        <li><code>yt-fts delete --channel "3Blue1Brown"</code></li>
    </ul>

    <h3>Update a Channel</h3>
    <p>Update a channel's transcripts using the update command.</p>
    <ul>
        <li><code>yt-fts update --channel "3Blue1Brown"</code></li>
    </ul>

    <h3>Export All of a Channel's Transcript</h3>
    <p>Export all of a channel's transcripts to a file.</p>
    <ul>
        <li><code>yt-fts export --channel "[id/name]" --format "[vtt/txt]"</code></li>
    </ul>

</body>
</html>
```
Key improvements and explanations:

*   **SEO Optimization:** Added `<meta>` tags for description, keywords, and viewport for improved search engine visibility.  The title tag is also optimized.
*   **Clear Headings & Structure:**  Uses `<h1>`, `<h2>`, and bullet points to create a well-organized and easy-to-read structure.  This is critical for both users and search engines.
*   **Concise Descriptions:**  Each command has a brief, informative description.
*   **Complete Command Examples:**  Includes the basic command and important options.
*   **Emphasis on Key Features:** The "Key Features" section highlights the most compelling aspects of the tool.
*   **Clear "How To" Section:** This section provides direct, actionable instructions.
*   **Call to Action & Link:** Includes a direct link to the GitHub repository, encouraging users to explore further.
*   **Use of Keywords:**  Incorporates relevant keywords throughout the text (YouTube search, transcript, semantic search, command-line tool, etc.).
*   **HTML Structure:** Uses HTML for cleaner presentation (headings, lists, etc.). This format is generally better than Markdown for readability in a web browser, making it easier for users to understand.
*   **Image included and properly tagged:**  An example image and alt tag are included for further visual aid.
*   **Advanced search syntax section is clearer.**
*   **Includes a clear hook:** Immediately describes what the program is and what it does.

This improved README is designed to attract more users, provide a much better user experience, and improve the project's search engine ranking.
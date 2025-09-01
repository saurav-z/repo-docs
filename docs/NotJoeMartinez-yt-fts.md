html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>yt-fts: YouTube Full Text Search</title>
    <meta name="description" content="yt-fts is a command-line tool for searching YouTube video transcripts, enabling full-text search and semantic search capabilities.">
    <meta name="keywords" content="YouTube, search, full-text search, subtitles, transcripts, command-line tool, semantic search, OpenAI, Gemini">
</head>
<body>
    <h1>yt-fts: Unleash the Power of YouTube Video Search</h1>
    <p>Quickly and efficiently search the full text of YouTube video transcripts with the power of `yt-fts`, your go-to command-line tool.</p>

    <img src="https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14" alt="yt-fts Demo" width="500">

    <h2>Key Features</h2>
    <ul>
        <li><b>Full-Text Search:</b> Quickly find videos by searching keywords or phrases within their transcripts.</li>
        <li><b>Semantic Search:</b> Leverage the power of OpenAI or Gemini embeddings for context-aware search results.</li>
        <li><b>Channel & Playlist Support:</b> Download and search transcripts from entire YouTube channels or playlists.</li>
        <li><b>LLM Integration:</b> Chat with your data from YouTube channels using an LLM chatbot.</li>
        <li><b>Video Summarization:</b> Generate concise summaries of YouTube videos.</li>
        <li><b>Export Options:</b> Export search results and transcripts in various formats (CSV, TXT, VTT).</li>
        <li><b>Advanced Search Syntax:</b> Includes support for AND/OR searches and wildcards.</li>
    </ul>

    <h2>Getting Started</h2>

    <h3>Installation</h3>
    <p>Install yt-fts using pip:</p>
    <pre><code>pip install yt-fts</code></pre>

    <h3>Commands</h3>

    <h4><code>download</code> - Download Subtitles</h4>
    <p>Downloads subtitles for a channel or playlist.</p>
    <pre>
        <code>yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"</code><br>
        <code>yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"</code>
    </pre>
    <ul>
        <li><code>--playlist</code>: Download all videos from a playlist</li>
        <li><code>--language</code>: Language of the subtitles to download (default: en)</li>
        <li><code>--jobs</code>: Number of parallel download jobs (default: 8, recommended: 4-16)</li>
        <li><code>--cookies-from-browser</code>: Browser to extract cookies from (chrome, firefox, etc.)</li>
    </ul>

    <h4><code>diagnose</code> - Troubleshoot Download Issues</h4>
    <p>Tests your connection and provides recommendations for fixing common issues.</p>
    <pre>
        <code>yt-fts diagnose</code><br>
        <code>yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox</code>
    </pre>
    <ul>
        <li><code>--test-url</code>: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)</li>
        <li><code>--cookies-from-browser</code>: Browser to extract cookies from</li>
        <li><code>--jobs</code>: Number of parallel download jobs to test with (default: 8)</li>
    </ul>

    <h4><code>list</code> - View Library Content</h4>
    <p>Lists saved channels, videos, and transcripts.</p>
    <pre>
        <code>yt-fts list</code><br>
        <code>yt-fts list --channel "3Blue1Brown"</code>
    </pre>
    <ul>
        <li><code>--transcript</code>: Show transcript for a video</li>
        <li><code>--channel</code>: Show list of videos for a channel</li>
        <li><code>--library</code>: Show list of channels in library</li>
    </ul>

    <h4><code>update</code> - Update Subtitles</h4>
    <p>Updates subtitles for all or a specific channel.</p>
    <pre>
        <code>yt-fts update</code><br>
        <code>yt-fts update --channel "3Blue1Brown" --jobs 5</code>
    </pre>
    <ul>
        <li><code>--channel</code>: The name or id of the channel to update</li>
        <li><code>--language</code>: Language of the subtitles to download (default: en)</li>
        <li><code>--jobs</code>: Number of parallel download jobs (default: 8)</li>
        <li><code>--cookies-from-browser</code>: Browser to extract cookies from</li>
    </ul>

    <h4><code>delete</code> - Delete a Channel</h4>
    <p>Deletes a channel and its data.  Requires confirmation.</p>
    <pre>
        <code>yt-fts delete --channel "3Blue1Brown"</code>
    </pre>
    <ul>
        <li><code>--channel</code>: The name or id of the channel to delete (required)</li>
    </ul>

    <h4><code>export</code> - Export Transcripts</h4>
    <p>Exports transcripts for a channel in various formats.</p>
    <pre>
        <code>yt-fts export --channel "3Blue1Brown" --format txt</code><br>
        <code>yt-fts export --channel "3Blue1Brown" --format vtt</code>
    </pre>
    <ul>
        <li><code>--channel</code>: The name or id of the channel to export transcripts for (required)</li>
        <li><code>--format</code>: The format to export transcripts to. Supported formats: txt, vtt (default: txt)</li>
    </ul>

    <h4><code>search</code> - Full Text Search</h4>
    <p>Performs a full-text search within saved transcripts.</p>
    <pre>
        <code>yt-fts search "[search query]"</code><br>
        <code>yt-fts search "[search query]" --channel "[channel name or id]"</code>
    </pre>
    <ul>
        <li><code>--channel</code>: The name or id of the channel to search in</li>
        <li><code>--video-id</code>: The id of the video to search in</li>
        <li><code>--limit</code>: Number of results to return (default: 10)</li>
        <li><code>--export</code>: Export search results to a CSV file</li>
    </ul>
    <p><b>Advanced Search Syntax:</b>  Supports SQLite <a href="https://www.sqlite.org/fts3.html#full_text_index_queries">Enhanced Query Syntax</a> including prefix queries and boolean operators.</p>
    <pre>
        <code>yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"</code><br>
        <code>yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"</code>
    </pre>

    <h4><code>embeddings</code> - Enable Semantic Search</h4>
    <p>Generates embeddings for a channel, enabling semantic search.</p>
    <pre>
        <code>yt-fts embeddings --channel "3Blue1Brown"</code><br>
        <code>yt-fts embeddings --interval 60 --channel "3Blue1Brown"</code>
    </pre>
    <ul>
        <li><code>--channel</code>: The name or id of the channel to generate embeddings for</li>
        <li><code>--api-key</code>: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)</li>
        <li><code>--interval</code>: Interval in seconds to split the transcripts into chunks (default: 30)</li>
    </ul>
    <p>Requires an OpenAI or Gemini API key.  Set <code>OPENAI_API_KEY</code> or <code>GEMINI_API_KEY</code> environment variable.</p>

    <h4><code>vsearch</code> - Semantic Search</h4>
    <p>Performs a semantic search using generated embeddings (requires <code>embeddings</code> to be run first).</p>
    <pre>
        <code>yt-fts vsearch "[search query]" --channel "[channel name or id]"</code>
    </pre>
    <ul>
        <li><code>--channel</code>: The name or id of the channel to search in</li>
        <li><code>--video-id</code>: The id of the video to search in</li>
        <li><code>--limit</code>: Number of results to return (default: 10)</li>
        <li><code>--export</code>: Export search results to a CSV file</li>
        <li><code>--api-key</code>: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)</li>
    </ul>

    <h4><code>llm</code> - Chat Bot</h4>
    <p>Starts an interactive chat session using semantic search results as context.</p>
    <pre>
        <code>yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"</code>
    </pre>
    <ul>
        <li><code>--channel</code>: The name or id of the channel to use (required)</li>
        <li><code>--api-key</code>: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)</li>
    </ul>

    <h4><code>summarize</code> - Summarize Videos</h4>
    <p>Generates summaries for YouTube videos.</p>
    <pre>
        <code>yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"</code><br>
        <code>yt-fts summarize "9-Jl0dxWQs8"</code>
    </pre>
    <ul>
        <li><code>--model, -m</code>: Model to use in summary</li>
        <li><code>--api-key</code>: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)</li>
    </ul>
    <p>Example Output:</p>
    <pre>
        <code>In this video, 3Blue1Brown explores how large language models (LLMs) like GPT-3 might store facts within their vast...</code>
        <code>1 Introduction to Fact Storage in LLMs:</code>
        <code>• The video starts by questioning how LLMs store specific facts and</code>
        <code>  introduces the idea that these facts might be stored in a particular part of the</code>
        <code>  network known as multi-layer perceptrons (MLPs).</code>
        <code>• 0:00</code>
    </pre>

    <h4><code>config</code> - View Configuration</h4>
    <p>Displays current configuration settings.</p>
    <pre>
        <code>yt-fts config</code>
    </pre>

    <h2>How To</h2>

    <h3>Export Search Results</h3>
    <p>Export your search results to a CSV file using the <code>--export</code> flag with either the <code>search</code> or <code>vsearch</code> commands.</p>
    <pre>
        <code>yt-fts search "life in the big city" --export</code><br>
        <code>yt-fts vsearch "existing in large metropolaten center" --export</code>
    </pre>

    <h3>Delete a Channel</h3>
    <p>Use the <code>delete</code> command to remove a channel and its associated data.</p>
    <pre>
        <code>yt-fts delete --channel "3Blue1Brown"</code>
    </pre>

    <h3>Update a Channel</h3>
    <p>Update the full-text search index for a channel using the <code>update</code> command.</p>
    <pre>
        <code>yt-fts update --channel "3Blue1Brown"</code>
    </pre>

    <h3>Export Transcript of a Channel</h3>
    <p>Export the entire transcript of a channel in the specified format.</p>
    <pre>
        <code>yt-fts export --channel "[id/name]" --format "[vtt/txt]"</code>
    </pre>

    <h2>Contribute</h2>
    <p>Find the source code and contribute on GitHub: <a href="https://github.com/NotJoeMartinez/yt-fts">https://github.com/NotJoeMartinez/yt-fts</a></p>

</body>
</html>
```
Key improvements and explanations:

*   **SEO Optimization:**
    *   Includes a descriptive `<meta name="description">` tag with relevant keywords.
    *   Uses a more descriptive and keyword-rich title.
    *   Uses `<meta name="keywords">` to provide additional keywords for search engines.
    *   Uses headings (H1, H2, H3, etc.) to structure the content, improving readability for users and search engines.
*   **Clear Structure & Readability:**
    *   Uses bullet points for key features, making them easy to scan.
    *   Uses `<pre>` tags for code examples, preserving formatting.
    *   Organizes commands logically, with clear explanations and option details.
*   **Concise Summary & Hook:**
    *   The first paragraph clearly summarizes the tool's purpose.
    *   The opening sentence is designed as a concise "hook" to grab the user's attention.
*   **Comprehensive Content:**
    *   Covers all the commands and options from the original README.
    *   Includes examples for each command.
    *   Adds "How To" sections with practical use cases.
*   **Emphasis on Semantic Search:**
    *   Highlights the semantic search features, a key selling point.
    *   Clear instructions on enabling semantic search.
*   **Contribution Link:** Provides a clear link back to the original GitHub repository.
*   **HTML Formatting:**  Uses HTML structure for better formatting and SEO, especially with headings, making the content more readable and search engine-friendly.  Includes alt text for the image.
*   **API Key Instructions:** Clearly states where to set the API key, and that you have to set either the OPENAI or GEMINI api key.
*   **Fixed Update command**: The update command has been corrected to show the the correct functionality.
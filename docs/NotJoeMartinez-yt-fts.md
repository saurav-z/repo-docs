html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>yt-fts: YouTube Full Text Search - Command-Line Tool</title>
    <meta name="description" content="yt-fts is a powerful command-line tool for searching and exploring YouTube video transcripts, enabling full-text search, semantic search, and more.">
    <meta name="keywords" content="YouTube, search, subtitles, transcripts, command-line, full-text search, semantic search, OpenAI, Gemini, chroma, yt-dlp">
    <!-- Add more meta tags for SEO as needed -->
</head>
<body>

    <h1>yt-fts: YouTube Full Text Search</h1>
    <p>Unleash the power of text search within YouTube videos with <code>yt-fts</code>, a command-line tool that lets you find specific moments and insights within your favorite channels.  <a href="https://github.com/NotJoeMartinez/yt-fts">Explore the repo!</a></p>

    <h2>Key Features</h2>
    <ul>
        <li><b>Full Text Search:</b> Search within YouTube video transcripts using keywords and phrases.</li>
        <li><b>Semantic Search:</b> Leverage OpenAI, Gemini or Chroma to search based on the meaning of your query.</li>
        <li><b>Transcript Downloading:</b> Easily download subtitles from YouTube channels and playlists.</li>
        <li><b>Interactive LLM Chatbot:</b> Engage in conversational Q&A with a chatbot that uses semantic search results as context.</li>
        <li><b>Video Summarization:</b> Generate concise summaries of YouTube videos.</li>
        <li><b>Flexible Exporting:</b> Export search results to CSV files and transcripts to text or VTT format.</li>
    </ul>

    <img src="https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14" alt="yt-fts Demo" width="500">

    <h2>Installation</h2>
    <p>Install <code>yt-fts</code> using pip:</p>
    <pre><code>pip install yt-fts</code></pre>

    <h2>Commands</h2>

    <h3><code>download</code> - Download Subtitles</h3>
    <p>Download subtitles for a YouTube channel or playlist.</p>
    <pre><code>yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-p, --playlist</code>: Download all videos from a playlist</li>
        <li><code>-l, --language</code>: Language of the subtitles (default: en)</li>
        <li><code>-j, --jobs</code>: Number of parallel download jobs (default: 8, recommended: 4-16)</li>
        <li><code>--cookies-from-browser</code>: Browser to extract cookies from (chrome, firefox, etc.)</li>
    </ul>

    <h3><code>diagnose</code> - Diagnose Download Issues</h3>
    <p>Test your connection to YouTube and get recommendations for fixing common issues.</p>
    <pre><code>yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-u, --test-url</code>: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)</li>
        <li><code>--cookies-from-browser</code>: Browser to extract cookies from</li>
        <li><code>-j, --jobs</code>: Number of parallel download jobs to test with (default: 8)</li>
    </ul>

    <h3><code>list</code> - List Saved Data</h3>
    <p>List saved channels, videos, and transcripts.  (ss) next to the channel name indicates semantic search is enabled.</p>
    <pre><code>yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
yt-fts list --library
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-t, --transcript</code>: Show transcript for a video</li>
        <li><code>-c, --channel</code>: Show list of videos for a channel</li>
        <li><code>-l, --library</code>: Show list of channels in library</li>
    </ul>

    <h3><code>update</code> - Update Subtitles</h3>
    <p>Update subtitles for all or a specific channel.</p>
    <pre><code>yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-c, --channel</code>: The name or id of the channel to update</li>
        <li><code>-l, --language</code>: Language of the subtitles to download (default: en)</li>
        <li><code>-j, --jobs</code>: Number of parallel download jobs (default: 8)</li>
        <li><code>--cookies-from-browser</code>: Browser to extract cookies from</li>
    </ul>

    <h3><code>delete</code> - Delete a Channel</h3>
    <p>Delete a channel and all its associated data.  Confirmation is required.</p>
    <pre><code>yt-fts delete --channel "3Blue1Brown"
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-c, --channel</code>: The name or id of the channel to delete (required)</li>
    </ul>

    <h3><code>export</code> - Export Transcripts</h3>
    <p>Export transcripts for a channel to a file.</p>
    <pre><code>yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-c, --channel</code>: The name or id of the channel to export transcripts for (required)</li>
        <li><code>-f, --format</code>: The format to export transcripts to. Supported formats: txt, vtt (default: txt)</li>
    </ul>

    <h3><code>search</code> - Full Text Search</h3>
    <p>Perform full-text searches within the saved transcripts.</p>
    <pre><code>yt-fts search "[search query]"
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "[search query]" --video-id "[video id]"
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts search "[search query]" --export --channel "[channel name or id]"
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-c, --channel</code>: The name or id of the channel to search in</li>
        <li><code>-v, --video-id</code>: The id of the video to search in</li>
        <li><code>-l, --limit</code>: Number of results to return (default: 10)</li>
        <li><code>-e, --export</code>: Export search results to a CSV file</li>
    </ul>
    <p><b>Advanced Search Syntax:</b></p>
    <p>Supports <a href="https://www.sqlite.org/fts3.html#full_text_index_queries">SQLite Enhanced Query Syntax</a>, including prefix queries, AND/OR operators, and wildcards.</p>
    <pre><code>yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
</code></pre>

    <h2>Semantic Search and RAG</h2>
    <p>Enable semantic search for a channel using OpenAI or Gemini API keys.</p>

    <h3><code>embeddings</code> - Generate Embeddings</h3>
    <p>Generates embeddings for a channel, enabling semantic search.</p>
    <pre><code># Set your API key in environment variables
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-c, --channel</code>: The name or id of the channel to generate embeddings for</li>
        <li><code>--api-key</code>: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)</li>
        <li><code>-i, --interval</code>: Interval in seconds to split the transcripts into chunks (default: 30)</li>
    </ul>

    <h3><code>vsearch</code> - Semantic (Vector) Search</h3>
    <p>Search using semantic similarity based on the generated embeddings.</p>
    <pre><code>yt-fts vsearch "[search query]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --video-id "[video id]"
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-c, --channel</code>: The name or id of the channel to search in</li>
        <li><code>-v, --video-id</code>: The id of the video to search in</li>
        <li><code>-l, --limit</code>: Number of results to return (default: 10)</li>
        <li><code>-e, --export</code>: Export search results to a CSV file</li>
        <li><code>--api-key</code>: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)</li>
    </ul>

    <h3><code>llm</code> - Interactive Chat Bot</h3>
    <p>Engage in a conversational Q&A session using the semantic search results as context.</p>
    <pre><code>yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>-c, --channel</code>: The name or id of the channel to use (required)</li>
        <li><code>--api-key</code>: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)</li>
    </ul>

    <h3><code>summarize</code> - Summarize Videos</h3>
    <p>Summarize a YouTube video transcript.</p>
    <pre><code>yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
yt-fts summarize "9-Jl0dxWQs8"
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
</code></pre>
    <p><b>Options:</b></p>
    <ul>
        <li><code>--model, -m</code>: Model to use in summary</li>
        <li><code>--api-key</code>: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)</li>
    </ul>

    <h3><code>config</code> - Show Configuration</h3>
    <p>Displays configuration settings, including database and Chroma paths.</p>
    <pre><code>yt-fts config
</code></pre>

    <h2>How To</h2>

    <h3>Export Search Results</h3>
    <p>Export search results to a CSV file using the <code>--export</code> flag with either the <code>search</code> or <code>vsearch</code> commands.</p>
    <pre><code>yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
</code></pre>

    <h3>Delete a Channel</h3>
    <p>Delete a channel and its data using the <code>delete</code> command.</p>
    <pre><code>yt-fts delete --channel "3Blue1Brown"
</code></pre>

    <h3>Update a Channel</h3>
    <p>Update a channel using the <code>update</code> command. <i>Note:</i> Currently only updates full text search data, not semantic embeddings.</p>
    <pre><code>yt-fts update --channel "3Blue1Brown"
</code></pre>

    <h3>Export a Channel's Transcript</h3>
    <p>Export a channel's transcript to a file using the <code>export</code> command.</p>
    <pre><code>yt-fts export --channel "[id/name]" --format "[vtt/txt]"
</code></pre>
</body>
</html>
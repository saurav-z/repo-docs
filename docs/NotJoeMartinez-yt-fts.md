# yt-fts: Unlock YouTube Insights with Powerful Full-Text Search

**Quickly search, analyze, and extract information from YouTube channel subtitles using yt-fts!**  (Original Repo: [https://github.com/NotJoeMartinez/yt-fts](https://github.com/NotJoeMartinez/yt-fts))

This command-line tool empowers you to delve deep into YouTube content, offering a range of features from basic keyword searches to advanced semantic analysis using AI.

## Key Features

*   **Full-Text Search:**  Search across all video subtitles within a channel or playlist for specific keywords or phrases.
*   **Semantic Search:**  Leverage AI-powered semantic search (using OpenAI or Gemini embeddings) for more nuanced and relevant results.
*   **LLM Chat Bot:**  Engage in conversational exploration of channel content with an integrated LLM chatbot.
*   **Video Summarization:** Generate concise summaries of YouTube videos with timestamped links.
*   **Flexible Data Management:** Download, update, list, and delete channel data with ease.
*   **Data Export:**  Export search results and transcripts in various formats for further analysis.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Command-Line Usage

### Download Subtitles

Download subtitles for a channel or playlist and store them in a searchable database.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
```

*   `--playlist`:  Download all videos from a playlist
*   `--language`:  Specify the subtitle language (default: `en`)
*   `--jobs`:  Set the number of parallel download jobs (recommended: 4-16)
*   `--cookies-from-browser`: Use cookies from your browser to bypass login restrictions (e.g., `firefox`, `chrome`).

### Diagnose Download Issues

Test your connection to YouTube and troubleshoot common download problems.

```bash
yt-fts diagnose
```

*   `--test-url`:  Specify a URL to test.
*   `--cookies-from-browser`: Use cookies from your browser.
*   `--jobs`: Set the number of parallel download jobs to test.

### List Data

View your saved channels, videos, and transcripts.

```bash
yt-fts list
```

*   `--transcript`: Display the transcript for a specific video.
*   `--channel`: List videos for a specific channel.
*   `--library`: List channels in your library.

### Update Subtitles

Update the subtitles for existing channels in your library.

```bash
yt-fts update
```

*   `--channel`:  Update a specific channel.
*   `--language`:  Specify the subtitle language.
*   `--jobs`: Set the number of parallel download jobs.
*   `--cookies-from-browser`: Use cookies from your browser.

### Delete Channels

Delete a channel and its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

*   `--channel`:  The name or ID of the channel to delete (required).

### Export Transcripts

Export transcripts for a channel in various formats.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
```

*   `--channel`:  The name or ID of the channel to export transcripts for (required).
*   `--format`:  The export format: `txt` or `vtt` (default: `txt`).

### Full Text Search (`search`)

Perform full-text searches within channel subtitles.

```bash
yt-fts search "search query" --channel "3Blue1Brown"
```

*   `--channel`: Search within a specific channel.
*   `--video-id`: Search within a specific video.
*   `--limit`: Limit the number of results.
*   `--export`: Export results to a CSV file.

**Advanced Search Syntax:** Utilize SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for more advanced search capabilities, including AND/OR operators and wildcards.

### Semantic Search and RAG

Enable semantic search using OpenAI or Gemini API keys. Set your API key using the environment variable `OPENAI_API_KEY` or `GEMINI_API_KEY` or pass the key with the `--api-key` flag.

### Embeddings

Generate embeddings for a specified channel, enabling semantic search.

```bash
yt-fts embeddings --channel "3Blue1Brown"
```

*   `--channel`:  The name or ID of the channel to generate embeddings for.
*   `--api-key`:  Your OpenAI or Gemini API key.
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

### Vector Search (`vsearch`)

Search using semantic similarity, requiring that you enable semantic search for a channel with `embeddings`.

```bash
yt-fts vsearch "search query" --channel "3Blue1Brown"
```

*   Same options as `search`.

### LLM Chat Bot (`llm`)

Engage in an interactive chat session, using the semantic search results as context for your questions.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

*   `--channel`:  The name or ID of the channel (required).
*   `--api-key`: Your OpenAI or Gemini API key.

### Summarize Video Transcripts (`summarize`)

Generate concise summaries of YouTube videos.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
```

*   `--model, -m`:  Specify the model to use (e.g., "gpt-3.5-turbo").
*   `--api-key`: Your OpenAI or Gemini API key.

### Configuration (`config`)

View your configuration settings.

```bash
yt-fts config
```
## How To:

*   **Export search results:** Use the `--export` flag with `search` and `vsearch` commands to save the results to a CSV file.

```bash
yt-fts search "life in the big city" --export
```

*   **Delete a channel:** Use the `delete` command to remove a channel and its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

*   **Update a channel:**  Use the `update` command to refresh the subtitles.  Note this command currently does not update the semantic search embeddings.

```bash
yt-fts update --channel "3Blue1Brown"
```

*   **Export all of a channel's transcript:**

```bash
yt-fts export --channel "[channel name/id]" --format "[vtt/txt]"
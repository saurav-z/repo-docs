# yt-fts: Unleash the Power of YouTube Video Search (and More!)

**yt-fts** empowers you to search the full text of YouTube video subtitles, turning passive viewing into an interactive research experience. [View the project on GitHub](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   🔍 **Full-Text Search:** Quickly find specific keywords and phrases within any YouTube channel's subtitles.
*   💬 **Semantic Search:** Leverage the power of AI (OpenAI, Gemini, or ChromaDB) for advanced semantic understanding and search.
*   💾 **Database-Driven:** Stores subtitles in a searchable SQLite database for fast and efficient retrieval.
*   🔄 **Automated Updates:** Easily update your library with new videos and subtitles.
*   📤 **Export Transcripts:** Export transcripts in various formats (TXT, VTT) for offline use.
*   🤖 **LLM Chat Bot:** Engage in interactive Q&A sessions using channel content with an LLM model.
*   📝 **Video Summarization:** Quickly generate summaries of YouTube videos, including time-stamped links.
*   🛠️ **Troubleshooting:** Built-in diagnostic tools to resolve common download issues.

## Installation

Install `yt-fts` using `pip`:

```bash
pip install yt-fts
```

## Command Reference

### `download`

Downloads subtitles for a channel or playlist.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Use browser cookies (chrome, firefox, etc.) for authentication.

### `diagnose`

Tests your connection to YouTube and provides solutions to common errors.

```bash
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list`

Lists saved channels, videos, and transcripts.

```bash
yt-fts list --channel "3Blue1Brown"
```

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show videos for a specific channel
*   `-l, --library`: Show the list of channels

### `update`

Updates subtitles for all channels or a specific channel.

```bash
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel name or ID to update
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

### `delete`

Deletes a channel and all associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID to delete (required)

### `export`

Exports transcripts for a channel.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
```

**Options:**

*   `-c, --channel`: Channel name or ID to export (required)
*   `-f, --format`: Export format: `txt`, `vtt` (default: `txt`)

### `search` (Full Text Search)

Searches for a string within saved subtitles.

```bash
yt-fts search "search query" --channel "channel name or id"
```

**Options:**

*   `-c, --channel`: Channel name or ID to search within
*   `-v, --video-id`: Video ID to search within
*   `-l, --limit`: Maximum results to return (default: 10)
*   `-e, --export`: Export results to CSV

**Advanced Search Syntax:**  Supports SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries)

### `embeddings` (Semantic Search Setup)

Generates embeddings for semantic search functionality. Requires an OpenAI or Gemini API key.

```bash
# Set OpenAI API key:
# export OPENAI_API_KEY="your_api_key"
yt-fts embeddings --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID
*   `--api-key`: API key (or uses `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variables)
*   `-i, --interval`: Chunking interval in seconds (default: 30)

### `vsearch` (Semantic Search)

Performs semantic (vector-based) searches within a channel. Requires embeddings.

```bash
yt-fts vsearch "search query" --channel "channel name or id"
```

**Options:**

*   `-c, --channel`: Channel name or ID to search within
*   `-v, --video-id`: Video ID to search within
*   `-l, --limit`: Maximum results to return (default: 10)
*   `-e, --export`: Export results to CSV
*   `--api-key`: API key (or uses `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variables)

### `llm` (Chat Bot)

Starts an interactive chat session, using semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required)
*   `--api-key`: API key (or uses `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variables)

### `summarize`

Summarizes a YouTube video, providing time-stamped links.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (or uses `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variables)

### `config`

Shows current configuration settings (database and Chroma paths).

```bash
yt-fts config
```

## How To

### Export Search Results

Both `search` and `vsearch` support exporting results to a CSV file using the `--export` flag:

```bash
yt-fts search "your search term" --export
yt-fts vsearch "your semantic search term" --export
```

### Delete a Channel

Use the `delete` command to remove a channel:

```bash
yt-fts delete --channel "3Blue1Brown"
```

### Update a Channel

Use the `update` command to fetch the latest subtitles (Note: Currently, only full text search updates).

```bash
yt-fts update --channel "3Blue1Brown"
```

### Export All Transcripts

Export an entire channel's transcripts:

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
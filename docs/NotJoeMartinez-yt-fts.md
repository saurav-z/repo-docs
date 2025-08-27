# yt-fts: Unleash the Power of YouTube Full Text Search

**yt-fts is a command-line tool that allows you to search within YouTube video transcripts, turning your favorite channels into searchable knowledge bases.** ([Original Repo](https://github.com/NotJoeMartinez/yt-fts))

[![yt-fts demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

**Key Features:**

*   **Full-Text Search:** Quickly find specific keywords or phrases within YouTube video transcripts.
*   **Semantic Search:** Leverage the power of AI embeddings (OpenAI or Gemini) for more intelligent and context-aware searches.
*   **LLM Integration:** Interact with an LLM chatbot that uses semantic search results for context, enabling deeper exploration of channel content.
*   **Video Summarization:** Generate concise summaries of YouTube videos with timestamped URLs.
*   **Channel Management:** Easily download, update, list, delete, and export transcripts for your favorite channels.
*   **Advanced Search Syntax:** Utilize SQLite's Enhanced Query Syntax for powerful and flexible search queries (AND, OR, wildcards, etc.).

## Installation

Install yt-fts using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles

Downloads subtitles for a YouTube channel or playlist and stores them in a searchable database.  Use the `--cookies-from-browser` flag if you're getting errors.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
```

**Options:**
*   `-p, --playlist`: Download playlist
*   `-l, --language`: Subtitle language (default: en)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Use cookies from a browser (chrome, firefox, etc.)

### `diagnose` - Diagnose Download Issues

Tests your connection to YouTube and provides recommendations for fixing download errors.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**
*   `-u, --test-url`: URL to test
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Parallel download jobs (default: 8)

### `list` - List Saved Content

Lists saved channels, videos, and transcripts.

```bash
yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**
*   `-t, --transcript`: Show a video's transcript
*   `-c, --channel`: Show videos for a specific channel
*   `-l, --library`: Show all channels

### `update` - Update Subtitles

Updates subtitles for all or specific channels.

```bash
yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**
*   `-c, --channel`: Channel name or ID to update
*   `-l, --language`: Subtitle language (default: en)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Use cookies from a browser

### `delete` - Delete a Channel

Deletes a channel and all associated data from the database.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**
*   `-c, --channel`: Channel name or ID to delete (required)

### `export` - Export Transcripts

Exports video transcripts for a channel in various formats.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**
*   `-c, --channel`: Channel name or ID to export (required)
*   `-f, --format`: Export format (txt, vtt, default: txt)

### `search` - Full Text Search

Searches for a string in saved transcripts. Supports advanced search syntax.

```bash
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
```

**Options:**
*   `-c, --channel`: Channel name or ID to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export results to CSV

### `embeddings` - Semantic Search Setup

Generates embeddings for a channel, enabling semantic search and LLM integration. Requires an OpenAI or Gemini API key (set via environment variable or `--api-key` flag).

```bash
yt-fts embeddings --channel "3Blue1Brown"
```

**Options:**
*   `-c, --channel`: Channel name or ID to generate embeddings for
*   `--api-key`: API key
*   `-i, --interval`: Interval in seconds to split transcripts into chunks (default: 30)

### `vsearch` - Semantic Search

Performs a semantic search using AI embeddings.  Requires embeddings to be generated for the channel first.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
```

**Options:**
*   `-c, --channel`: Channel name or ID to search in
*   `-v, --video-id`: Video ID to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export results to CSV
*   `--api-key`: API key

### `llm` - LLM Chat Bot

Starts an interactive chat session using the semantic search results of your initial prompt as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**
*   `-c, --channel`: Channel name or ID (required)
*   `--api-key`: API key

### `summarize` - Summarize Video

Summarizes a YouTube video transcript.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
yt-fts summarize "9-Jl0dxWQs8"
```

**Options:**
*   `--model, -m`: Model to use in summary
*   `--api-key`: API key

### `config` - Show Configuration

Displays the current configuration settings, including database and ChromaDB paths.

```bash
yt-fts config
```

## How To

**Export Search Results:** Use the `--export` flag with `search` or `vsearch` to save results to a CSV file.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:** Delete a channel using the `delete` command:

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:** Use the `update` command to refresh channel subtitles.

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export all of a Channel's Transcript:** Export transcripts using the `export` command:

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
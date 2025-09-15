# yt-fts: Unleash the Power of YouTube Full Text Search 

**yt-fts** empowers you to search within YouTube video transcripts using a powerful command-line interface, enabling you to quickly find specific moments in your favorite channels.  [View the original repo](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

Key Features:

*   **Full Text Search:** Search YouTube video transcripts for keywords and phrases.
*   **Semantic Search:** Utilize advanced semantic search to find videos based on meaning using OpenAI or Gemini API integrations.
*   **Comprehensive Channel Management:** Download, update, list, and delete channel data with ease.
*   **Flexible Export Options:** Export search results and transcripts in various formats (CSV, TXT, VTT).
*   **LLM Integration:** Chat with an LLM using the semantic search results as context.
*   **Video Summarization:** Quickly generate summaries of YouTube video transcripts.
*   **Advanced Search Syntax:** Leverage SQLite's enhanced query syntax for powerful search capabilities.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles
Download subtitles for a YouTube channel or playlist. Supports parallel downloads and browser cookie import.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
```

**Options:**
*   `-p, --playlist`: Download playlist
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel jobs (default: 8)
*   `--cookies-from-browser`: Use browser cookies (e.g., `firefox`)

### `diagnose` - Diagnose Download Issues
Test YouTube connection and provide troubleshooting recommendations.

```bash
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**
*   `-u, --test-url`: Test URL
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Parallel jobs (default: 8)

### `list` - List Library Contents
List saved channels, videos, and transcripts.

```bash
yt-fts list --channel "3Blue1Brown"
```

**Options:**
*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

### `update` - Update Subtitles
Update subtitles for existing channels.

```bash
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**
*   `-c, --channel`: Channel to update
*   `-l, --language`: Subtitle language (default: `en`)
*   `-j, --jobs`: Parallel jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

### `delete` - Delete a Channel
Delete a channel and all associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**
*   `-c, --channel`: Channel to delete (required)

### `export` - Export Transcripts
Export transcripts in various formats (TXT, VTT).

```bash
yt-fts export --channel "3Blue1Brown" --format txt
```

**Options:**
*   `-c, --channel`: Channel to export (required)
*   `-f, --format`: Export format (txt, vtt, default: txt)

### `search` - Full Text Search
Perform full-text searches within your saved transcripts.  Utilizes SQLite's enhanced query syntax.

```bash
yt-fts search "[search query]" --channel "[channel name or id]"
```

**Options:**
*   `-c, --channel`: Channel to search
*   `-v, --video-id`: Search in specific video
*   `-l, --limit`: Results limit (default: 10)
*   `-e, --export`: Export results to CSV

**Advanced Search Syntax:**
Utilize SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries), including prefix queries and boolean operators.

```bash
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### Semantic Search and RAG

Enable semantic search using OpenAI or Gemini API keys. Set your API key in the environment variables `OPENAI_API_KEY` or `GEMINI_API_KEY`, or pass it directly with the `--api-key` flag.

### `embeddings` - Generate Embeddings
Generates embeddings for a channel, enabling semantic search.

```bash
yt-fts embeddings --channel "3Blue1Brown"
```

**Options:**
*   `-c, --channel`: Channel for embeddings
*   `--api-key`: API key (overrides environment variable)
*   `-i, --interval`: Text chunk interval (default: 30 seconds)

### `vsearch` - Semantic Search
Perform semantic searches based on meaning.  Requires embeddings to be generated.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
```

**Options:**
*   `-c, --channel`: Channel to search
*   `-v, --video-id`: Search in specific video
*   `-l, --limit`: Results limit (default: 10)
*   `-e, --export`: Export results to CSV
*   `--api-key`: API key (overrides environment variable)

### `llm` - Interactive Chat Bot
Engage in conversational Q&A using semantic search results.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**
*   `-c, --channel`: Channel to use (required)
*   `--api-key`: API key (overrides environment variable)

### `summarize` - Summarize Video Transcripts
Generate summaries of YouTube video transcripts.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
```

**Options:**
*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (overrides environment variable)

### `config` - Show Configuration
Display configuration settings.

```bash
yt-fts config
```

## How To

**Export search results:**

Export the results of `search` or `vsearch` to a CSV file using the `--export` flag.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a channel:**

Delete a channel with the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a channel:**

Update a channel's subtitles using the `update` command.

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export all of a channel's transcript:**

Export a channel's transcripts using the `export` command.

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
# yt-fts: Unleash the Power of YouTube Content with Full Text Search

**Quickly and easily search through YouTube video transcripts, powered by advanced search and AI features.** [Check out the original repo](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full Text Search:** Search YouTube transcripts using keywords and phrases with enhanced search syntax.
*   **Semantic Search:** Leverage OpenAI or Gemini embeddings for more relevant results.
*   **LLM Chat Bot:** Engage in interactive Q&A sessions using semantic search results as context.
*   **Video Summarization:** Generate concise summaries of YouTube videos with timestamped links.
*   **Efficient Data Management:** Download, update, and manage transcripts for multiple channels.
*   **Flexible Exporting:** Export search results and transcripts in various formats (CSV, TXT, VTT).
*   **Easy Installation:** Simple pip install for quick setup.

## Installation

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles

Download subtitles for a channel or playlist. Use `--cookies-from-browser` for access to videos that require a sign-in.

```bash
# Download a channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"

# Download a playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download playlist videos
*   `-l, --language`: Subtitle language (default: en)
*   `-j, --jobs`: Parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser for cookie extraction

### `diagnose` - Diagnose Download Issues

Identify and troubleshoot common download errors.

```bash
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test
*   `--cookies-from-browser`: Browser for cookie extraction
*   `-j, --jobs`: Number of parallel jobs

### `list` - List Saved Content

View your library of channels, videos, and transcripts.

```bash
# List all channels
yt-fts list

# List videos for a specific channel
yt-fts list --channel "3Blue1Brown"

# Show transcript for a specific video
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**

*   `-t, --transcript`: Show video transcript
*   `-c, --channel`: Show videos for a channel
*   `-l, --library`: Show list of channels

### `update` - Update Subtitles

Update subtitles for existing channels.

```bash
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel to update
*   `-l, --language`: Subtitle language (default: en)
*   `-j, --jobs`: Parallel download jobs
*   `--cookies-from-browser`: Browser for cookie extraction

### `delete` - Delete a Channel

Remove a channel and its data from the database.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required)

### `export` - Export Transcripts

Export transcripts in various formats.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
```

**Options:**

*   `-c, --channel`: Channel to export
*   `-f, --format`: Export format (txt, vtt)

### `search` - Full Text Search

Perform full-text searches within downloaded transcripts.

```bash
yt-fts search "[search query]" --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel to search
*   `-v, --video-id`: Video ID to search
*   `-l, --limit`: Number of results to return
*   `-e, --export`: Export results to CSV

**Advanced Search Syntax:**

Utilize SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for advanced search.

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show" 

# OR SEARCH 
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show" 

# wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show" 
```

### `embeddings` - Semantic Search Initialization

Generate embeddings for semantic search using OpenAI or Gemini. Requires setting `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable, or passing the key directly with `--api-key`.

```bash
yt-fts embeddings --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel to generate embeddings for
*   `--api-key`: OpenAI or Gemini API key
*   `-i, --interval`: Transcript chunking interval in seconds (default: 30)

### `vsearch` - Semantic Search (Vector Search)

Perform semantic searches based on OpenAI/Gemini embeddings.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel to search
*   `-v, --video-id`: Video ID to search
*   `-l, --limit`: Number of results
*   `-e, --export`: Export results to CSV
*   `--api-key`: OpenAI or Gemini API key

### `llm` - LLM Chat Bot

Engage in an interactive chat session using semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel to use (required)
*   `--api-key`: OpenAI or Gemini API key

### `summarize` - Video Summarization

Get concise summaries of YouTube videos.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: LLM model for summarization
*   `--api-key`: OpenAI or Gemini API key

### `config` - Show Configuration

View the current configuration, including database and chroma paths.

```bash
yt-fts config
```

## How To Guides

### Export Search Results

Use the `--export` flag to export search results to a CSV file.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

### Delete a Channel

Remove a channel with its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

### Update a Channel

The update command updates subtitles for full text search; semantic search embeddings are not updated with this command.

```bash
yt-fts update --channel "3Blue1Brown"
```

### Export All Transcripts

Create a directory with the channel ID and export transcripts.

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
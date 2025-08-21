# yt-fts: Unleash the Power of YouTube Search 

Tired of endlessly scrolling through YouTube? **yt-fts** is a powerful command-line tool that lets you instantly search the *full text* of YouTube video subtitles, uncovering hidden knowledge and saving you valuable time. ([View the Original Repo](https://github.com/NotJoeMartinez/yt-fts))

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

## Key Features

*   **Full-Text Search:** Quickly find specific keywords or phrases within YouTube video transcripts.
*   **Semantic Search:** Leverage the power of OpenAI, Gemini, or ChromaDB embeddings for more relevant search results.
*   **Advanced Search:** Utilize SQLite's Enhanced Query Syntax for precise searches with AND, OR, and wildcard options.
*   **Channel Management:** Download, update, list, and delete channels with ease.
*   **Interactive LLM Chatbot:** Engage in conversations with an LLM, using semantic search results as context.
*   **Video Summarization:** Get concise summaries of YouTube videos, complete with time-stamped links.
*   **Export Functionality:** Export search results and transcripts for analysis and sharing.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`

Download subtitles for a YouTube channel or playlist.

*   Takes a channel or playlist URL as input.
*   Use `--jobs` to specify the number of parallel download jobs for faster processing.
*   Use `--cookies-from-browser` to authenticate using your browser's cookies.

```bash
# Download a channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"

# Download a playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download playlist videos.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel jobs (default: 8, recommended: 4-16).
*   `--cookies-from-browser`: Use cookies from a specified browser (chrome, firefox, etc.).

### `diagnose`

Diagnose and troubleshoot download issues.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
*   `--cookies-from-browser`: Specify a browser to use for cookies.
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8).

### `list`

List saved channels, videos, and transcripts.

```bash
# List all channels
yt-fts list

# List videos for a specific channel
yt-fts list --channel "3Blue1Brown"

# Show transcript for a specific video
yt-fts list --transcript "dQw4w9WgXcQ"

# Show library (same as default)
yt-fts list --library
```

**Options:**

*   `-t, --transcript`: Show the transcript for a video.
*   `-c, --channel`: Show the list of videos for a channel.
*   `-l, --library`: Show the list of channels in the library.

### `update`

Update subtitles for existing channels.

```bash
# Update all channels
yt-fts update

# Update a specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel name or ID to update.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel jobs (default: 8).
*   `--cookies-from-browser`: Specify browser for cookies.

### `delete`

Delete a channel and all its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID to delete (required).

### `export`

Export video transcripts.

```bash
# Export to TXT (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to VTT
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel name or ID to export (required).
*   `-f, --format`: Export format (txt, vtt; default: txt).

### `search` (Full Text Search)

Perform full-text searches within video transcripts.

*   Search strings are limited to 40 characters.

```bash
# Search all channels
yt-fts search "[search query]"

# Search a specific channel
yt-fts search "[search query]" --channel "[channel name or id]"

# Search a specific video
yt-fts search "[search query]" --video-id "[video id]"

# Limit results
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# Export results to CSV
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel name or ID to search within.
*   `-v, --video-id`: Video ID to search within.
*   `-l, --limit`: Number of results to return (default: 10).
*   `-e, --export`: Export results to CSV.

**Advanced Search Syntax:**

Use SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for advanced search capabilities like prefix queries.

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR search
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# Wildcards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### Semantic Search and RAG

Enable semantic search using OpenAI or Gemini API keys by setting the `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable, or by using the `--api-key` flag.

### `embeddings`

Generate embeddings for a channel to enable semantic search.

```bash
# Set API key (example)
# export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
# or
# export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

yt-fts embeddings --channel "3Blue1Brown"

# Specify interval to split text (default: 30 seconds)
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID.
*   `--api-key`: API key (if not provided, uses environment variables).
*   `-i, --interval`: Interval in seconds for splitting transcripts (default: 30).

After running this command, the channel will be marked with `(ss)` in the `list` command, and you can use the `vsearch` command.

### `vsearch` (Semantic Search)

Perform semantic (vector) search using embeddings.

```bash
# Search by channel name
yt-fts vsearch "[search query]" --channel "[channel name or id]"

# Search in specific video
yt-fts vsearch "[search query]" --video-id "[video id]"

# Limit results
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# Export results to CSV
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel name or ID to search in.
*   `-v, --video-id`: Video ID to search within.
*   `-l, --limit`: Number of results (default: 10).
*   `-e, --export`: Export results to CSV.
*   `--api-key`: API key (uses environment variables if not provided).

### `llm` (Chat Bot)

Start an interactive chatbot session using semantic search results as context.  Requires semantic search to be enabled for the specified channel.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `--api-key`: API key (uses environment variables if not provided).

### `summarize`

Summarize a YouTube video's transcript.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use a different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use for summarization.
*   `--api-key`: API key (uses environment variables if not provided).

### `config`

Show configuration settings, including database and ChromaDB paths.

```bash
yt-fts config
```

## How To

**Export Search Results:**

Export your `search` and `vsearch` results to a CSV file for analysis.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Remove a channel and its data using the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Update existing transcripts (only for full text search, not semantic embeddings).

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:**

Create a directory with the channel ID and export the transcript.

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
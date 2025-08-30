# yt-fts: Supercharge Your YouTube Research with Full-Text Search 🚀

Quickly and efficiently search through the transcripts of YouTube channels with `yt-fts`, a powerful command-line tool that lets you find exactly what you're looking for. [View the original repository on GitHub](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full-Text Search:** Search YouTube transcripts using keywords, phrases, and advanced search operators.
*   **Semantic Search (with OpenAI/Gemini):** Leverage semantic search to find videos based on meaning, not just keywords.
*   **LLM-Powered Chat Bot:** Engage in interactive conversations with the tool, using search results as context.
*   **Video Summarization:** Generate concise summaries of YouTube videos, complete with time-stamped links.
*   **Channel and Playlist Download:** Easily download subtitles for entire channels or playlists.
*   **Flexible Export Options:** Export search results and transcripts to CSV, TXT, or VTT formats.
*   **Robust Database:** Utilizes an SQLite database for efficient storage and retrieval of transcripts.
*   **Advanced Search Syntax:** Supports SQLite [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for powerful searching.
*   **Integrated Troubleshooting:** Includes diagnostic tools to help resolve download issues.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`

Download subtitles for a channel or playlist.

```bash
# Download a channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"

# Download a playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16)
*   `--cookies-from-browser`: Browser to extract cookies from (chrome, firefox, etc.)

### `diagnose`

Diagnose 403 errors and other download issues.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list`

List saved channels, videos, and transcripts.

```bash
# List all channels
yt-fts list

# List videos for a specific channel
yt-fts list --channel "3Blue1Brown"

# Show transcript for a specific video
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

### `update`

Update subtitles for all channels in the library or a specific channel.

```bash
# Update all channels
yt-fts update

# Update specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: The name or id of the channel to update
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

### `delete`

Delete a channel and all its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to delete (required)

### `export`

Export transcripts for a channel.

```bash
# Export to txt format (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to vtt format
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: The name or id of the channel to export transcripts for (required)
*   `-f, --format`: The format to export transcripts to. Supported formats: txt, vtt (default: txt)

### `search` (Full Text Search)

Perform a full-text search for a string in saved channels.

```bash
# Search in all channels
yt-fts search "[search query]"

# Search in a specific channel
yt-fts search "[search query]" --channel "[channel name or id]"

# Limit results
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# Export results to csv
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file

**Advanced Search Syntax:**

Use SQLite [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for more powerful queries:

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR SEARCH
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# Wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings` (Semantic Search)

Enable semantic search for a channel using OpenAI or Gemini.  Requires the `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable, or `--api-key`.

```bash
# Enable semantic search for a channel
yt-fts embeddings --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

### `vsearch` (Semantic Search)

Perform a vector search using the embeddings generated by `embeddings`.

```bash
# Search by channel name
yt-fts vsearch "[search query]" --channel "[channel name or id]"

# Limit results
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# Export results to csv
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `llm` (Chat Bot)

Start an interactive chat session, using semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize`

Summarize a YouTube video transcript.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `config`

Show config settings.

```bash
yt-fts config
```

## How To

**Export search results:**

For both the `search` and `vsearch` commands you can export the results to a csv file with the `--export` flag:

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a channel:**

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a channel:**

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export all of a channel's transcript:**

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
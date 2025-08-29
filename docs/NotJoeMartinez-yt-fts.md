# yt-fts: Unleash the Power of YouTube Search

**Quickly and efficiently search the full text of YouTube video transcripts with yt-fts, a powerful command-line tool.** ([See original repo](https://github.com/NotJoeMartinez/yt-fts))

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

## Key Features

*   **Full-Text Search:**  Search the full transcripts of YouTube channels using keywords or phrases.
*   **Semantic Search:**  Leverage OpenAI or Gemini's embeddings for advanced, meaning-based search.
*   **Time-Stamped Results:**  Quickly find the exact timestamps within videos where your search terms appear.
*   **Channel and Playlist Support:**  Download transcripts from entire channels or playlists.
*   **Flexible Export:**  Export search results to CSV or transcripts to TXT/VTT formats.
*   **LLM Chat Bot:**  Engage in interactive conversations with a chatbot using semantic search results as context.
*   **Video Summarization:** Generate concise summaries of YouTube video transcripts.
*   **Robust Installation & Updates:** Easy installation via pip and features to update and maintain your local database.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`

Download subtitles for a channel or playlist.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16)
*   `--cookies-from-browser`: Browser to extract cookies from (chrome, firefox, etc.)

### `diagnose`

Diagnose 403 errors and other download issues.

```bash
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list`

List saved channels, videos, and transcripts.

```bash
yt-fts list --channel "3Blue1Brown"
```

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

### `update`

Update subtitles for all channels in the library or a specific channel.

```bash
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
yt-fts export --channel "3Blue1Brown" --format txt
```

**Options:**

*   `-c, --channel`: The name or id of the channel to export transcripts for (required)
*   `-f, --format`: The format to export transcripts to. Supported formats: txt, vtt (default: txt)

### `search` (Full Text Search)

Full text search for a string in saved channels.

```bash
yt-fts search "[search query]" --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file

**Advanced Search Syntax:**

Supports sqlite [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries).

```bash
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
```

### `embeddings`

Fetches embeddings for specified channel

```bash
yt-fts embeddings --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

### `vsearch` (Semantic Search)

Semantic search for a string in saved channels, enabled by `embeddings` command.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `llm` (Chat Bot)

Starts interactive chat session with a model using the semantic search results.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize`

Summarizes a YouTube video transcript, providing time stamped URLS.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `config`

Show config settings including database and chroma paths.

```bash
yt-fts config
```

## How To

**Export search results:**

For both the `search` and `vsearch` commands you can export the results to a csv file with 
the `--export` flag.
```bash
yt-fts search "life in the big city" --export
```

**Delete a channel:**
You can delete a channel with the `delete` command. 

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a channel:**

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export all of a channel's transcript:**
```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
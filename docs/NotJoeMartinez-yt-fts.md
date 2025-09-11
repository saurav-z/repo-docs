# yt-fts: Supercharge Your YouTube Research with Full-Text Search

**Quickly search and analyze YouTube video transcripts using a powerful command-line tool.** ([View on GitHub](https://github.com/NotJoeMartinez/yt-fts))

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

**Key Features:**

*   **Full-Text Search:**  Find specific keywords and phrases within YouTube video transcripts.
*   **Semantic Search:** Leverage AI-powered semantic search using OpenAI or Gemini embeddings for more relevant results.
*   **Time-Stamped Results:** Get direct links to the exact moments in videos where your search terms appear.
*   **Channel & Playlist Support:** Download and search transcripts from entire channels or playlists.
*   **LLM Integration:**  Use LLMs to engage in a chat bot and get question-specific results for a channel.
*   **Video Summarization:** Generate concise summaries of YouTube video transcripts.
*   **Advanced Search Syntax:** Utilize SQLite's Enhanced Query Syntax for complex searches (AND, OR, wildcards).
*   **Flexible Export Options:** Export search results to CSV and transcripts to TXT or VTT formats.
*   **Easy Installation:** Simple `pip install yt-fts` installation.

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
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"

# Download playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16)
*   `--cookies-from-browser`: Browser to extract cookies from (chrome, firefox, etc.)

### `diagnose`

Diagnose and troubleshoot download issues.

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
yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
yt-fts list --library
```

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

### `update`

Update subtitles for existing channels in your library.

```bash
yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: The name or id of the channel to update
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

### `delete`

Delete a channel and all its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to delete (required)

### `export`

Export transcripts for a channel.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: The name or id of the channel to export transcripts for (required)
*   `-f, --format`: The format to export transcripts to. Supported formats: txt, vtt (default: txt)

### `search` (Full Text Search)

Perform a full-text search within your saved channels.

```bash
yt-fts search "[search query]"
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "[search query]" --video-id "[video id]"
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file

**Advanced Search Syntax:** Utilize SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for complex queries (e.g., "knife AND Malibu", "rea\* kni\* Mali\*").

### `embeddings`

Generate embeddings for semantic search, requires an OpenAI or Gemini API key.

```bash
# Set your API key:
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

### `vsearch` (Semantic Search)

Perform semantic search using OpenAI or Gemini embeddings.

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --video-id "[video id]"
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `llm` (Chat Bot)

Engage in a conversational search session with a channel. Requires semantic search to be enabled.

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
yt-fts summarize "9-Jl0dxWQs8"
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `config`

Show configuration settings.

```bash
yt-fts config
```

## How To

**Export Search Results:**

Use the `--export` flag with `search` or `vsearch` to export results to a CSV file.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Use the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Use the `update` command.

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export all of a channel's transcript:**

```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
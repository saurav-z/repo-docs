# yt-fts: Supercharge Your YouTube Research with Full-Text Search

Quickly search and analyze YouTube video transcripts using `yt-fts`, a powerful command-line tool that lets you find exactly what you need.  [Check out the original repo](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full-Text Search:** Find keywords and phrases within YouTube video transcripts.
*   **Semantic Search:**  Leverage OpenAI, Gemini, or ChromaDB for advanced, context-aware searches.
*   **LLM Integration:** Chat with an LLM (Large Language Model) that uses your search results as context, enabling in-depth analysis.
*   **Video Summarization:** Generate concise summaries of YouTube videos with time-stamped links.
*   **Batch Downloads:** Efficiently download subtitles for entire channels or playlists.
*   **Flexible Output:** Export search results to CSV and transcripts in various formats (TXT, VTT).

## Installation

Install `yt-fts` easily using pip:

```bash
pip install yt-fts
```

## Commands

### `download`

Download subtitles for a YouTube channel or playlist.

```bash
# Download channel
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

Diagnose and troubleshoot potential download issues.

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

# Show library (same as default)
yt-fts list --library
```

**Options:**

*   `-t, --transcript`: Show transcript for a video
*   `-c, --channel`: Show list of videos for a channel
*   `-l, --library`: Show list of channels in library

### `update`

Update subtitles for existing channels.

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

Delete a channel and its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to delete (required)

### `export`

Export video transcripts in various formats.

```bash
# Export to txt format (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to vtt format
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: The name or id of the channel to export transcripts for (required)
*   `-f, --format`: The format to export transcripts to. Supported formats: txt, vtt (default: txt)

### `search`

Perform full-text searches within downloaded transcripts. Supports SQLite's Enhanced Query Syntax for advanced searches.

```bash
# search in all channels
yt-fts search "[search query]" 

# search in channel 
yt-fts search "[search query]" --channel "[channel name or id]" 

# search in specific video
yt-fts search "[search query]" --video-id "[video id]"

# limit results 
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# export results to csv
yt-fts search "[search query]" --export --channel "[channel name or id]" 
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file

**Advanced Search Syntax:**

*   **AND search:**  `yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"`
*   **OR search:**  `yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"`
*   **Wildcards:** `yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"`

### `embeddings`

Generate embeddings for semantic search.  Requires an OpenAI or Gemini API key set in the environment variable `OPENAI_API_KEY` or `GEMINI_API_KEY`, or you can use the `--api-key` flag.

```bash
# Set API key in environment
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"

# Split transcripts into chunks
yt-fts embeddings --interval 60 --channel "3Blue1Brown" 
```

**Options:**

*   `-c, --channel`: The name or id of the channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

### `vsearch`

Perform semantic (vector) search using embeddings.

```bash
# search by channel name
yt-fts vsearch "[search query]" --channel "[channel name or id]"

# search in specific video
yt-fts vsearch "[search query]" --video-id "[video id]"

# limit results 
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# export results to csv
yt-fts vsearch "[search query]" --export --channel "[channel name or id]" 
```

**Options:**

*   `-c, --channel`: The name or id of the channel to search in
*   `-v, --video-id`: The id of the video to search in
*   `-l, --limit`: Number of results to return (default: 10)
*   `-e, --export`: Export search results to a CSV file
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `llm`

Engage in an interactive chat session with a model, leveraging semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize`

Generate summaries of YouTube video transcripts.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `config`

Display configuration settings, including database and ChromaDB paths.

```bash
yt-fts config
```

## How To

**Export Search Results:**

Export results to a CSV file:

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Delete a channel and its associated data:

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Update a channel's full-text search data. Note: does not update semantic search embeddings.

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export All of a Channel's Transcript:**

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
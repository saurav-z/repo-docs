# yt-fts: Supercharge Your YouTube Research with Full-Text Search

**Quickly search and analyze YouTube video transcripts with powerful command-line tools.** ([View the GitHub Repository](https://github.com/NotJoeMartinez/yt-fts))

<p align="center">
  <img src="https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14" alt="yt-fts demo" width="600">
</p>

## Key Features

*   **Full-Text Search:**  Find specific keywords and phrases within YouTube video transcripts.
*   **Semantic Search:** Leverage OpenAI or Gemini API embeddings for contextually relevant search results.
*   **Channel & Playlist Support:** Download and search transcripts from entire channels or playlists.
*   **Advanced Search:** Utilize SQLite's Enhanced Query Syntax for complex search queries (AND, OR, wildcards).
*   **LLM Integration:**  Chat with an LLM (Large Language Model) using semantic search results as context.
*   **Video Summarization:** Generate concise summaries of YouTube video transcripts.
*   **Flexible Export:** Export search results and transcripts in various formats (CSV, TXT, VTT).

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Command-Line Tools

### `download` - Download Subtitles

Download subtitles for a YouTube channel or playlist.  Use browser cookies to bypass login requirements.

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

### `diagnose` - Diagnose Download Issues

Diagnose and troubleshoot common download errors.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list` - List Library Contents

List saved channels, videos, and transcripts.  Channels with semantic search enabled are marked with (ss).

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

### `update` - Update Subtitles

Update subtitles for existing channels in your library.

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

### `delete` - Delete a Channel

Delete a channel and all associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to delete (required)

### `export` - Export Transcripts

Export transcripts for a channel in various formats.

```bash
# Export to txt format (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to vtt format
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: The name or id of the channel to export transcripts for (required)
*   `-f, --format`: The format to export transcripts to. Supported formats: txt, vtt (default: txt)

### `search` - Full-Text Search

Perform full-text searches within downloaded transcripts using a flexible syntax. Search strings limited to 40 characters.

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

Use SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for more advanced search capabilities:

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR SEARCH
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings` - Semantic Search (Requires API Key)

Generate embeddings for a channel using OpenAI or Gemini API for semantic search. Set `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable or use the `--api-key` flag.

```bash
# make sure API key is set
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"

# specify time interval in seconds to split text by default is 30
# the larger the interval the more accurate the llm response
# but semantic search will have more text for you to read.
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

### `vsearch` - Vector Search (Semantic Search)

Perform semantic (vector) search on channels with embeddings enabled.  Results are sorted by similarity.

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

### `llm` - LLM Chat Bot (Requires Semantic Search)

Start an interactive chat session with an LLM, leveraging semantic search results for context. Channel must have semantic search enabled.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize` - Video Summarization (Requires API Key)

Summarize YouTube video transcripts.

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

Example Output:

```
In this video, 3Blue1Brown explores how large language models (LLMs) like GPT-3
might store facts within their vast...

 1 Introduction to Fact Storage in LLMs:
    • The video starts by questioning how LLMs store specific facts and
      introduces the idea that these facts might be stored in a particular part of the
      network known as multi-layer perceptrons (MLPs).
    • 0:00
 2 Overview of Transformers and MLPs:
    • Provides a refresher on transformers and explains that the video will focus
```

### `config` - Show Configuration

Display configuration settings, including database and Chroma paths.

```bash
yt-fts config
```

## How To Guides

**Export Search Results:**

Export search results to a CSV file:

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export All Transcripts for a Channel:**

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
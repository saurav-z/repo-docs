# yt-fts: Supercharge Your YouTube Research with Full Text Search 🚀

Unleash the power of in-depth YouTube video analysis with `yt-fts`, a command-line tool that allows you to search, analyze, and summarize YouTube video transcripts. [Check out the original repo](https://github.com/NotJoeMartinez/yt-fts) for the latest updates.

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full-Text Search:** Quickly find specific keywords and phrases within YouTube video transcripts.
*   **Semantic Search:** Leverage OpenAI, Gemini, or ChromaDB embeddings for advanced context-aware search.
*   **LLM Integration:** Engage in interactive conversations and get answers powered by LLMs.
*   **Video Summarization:** Generate concise summaries of YouTube videos.
*   **Channel & Playlist Support:** Download and search transcripts from entire channels and playlists.
*   **Flexible Output:** Export search results to CSV and transcripts to TXT/VTT formats.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`: Download Subtitles

Download subtitles for a YouTube channel or playlist and populate your local database.

```bash
# Download a channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"

# Download a playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist
*   `-l, --language`: Language of subtitles (default: `en`)
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16)
*   `--cookies-from-browser`: Use cookies from your browser (e.g., `chrome`, `firefox`). Useful for channels requiring login.

### `diagnose`: Diagnose Download Issues

Troubleshoot download errors and identify potential problems.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`)
*   `--cookies-from-browser`: Browser to extract cookies from.
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8).

### `list`: List Saved Content

View saved channels, videos, and transcripts.

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

*   `-t, --transcript`: Show transcript for a video.
*   `-c, --channel`: Show a list of videos for a channel.
*   `-l, --library`: Show the list of channels in your library.

### `update`: Update Subtitles

Update the subtitles for all channels in the library or a specific channel. This command will attempt to download subtitles if they are added later.

```bash
# Update all channels
yt-fts update

# Update a specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel name or ID to update.
*   `-l, --language`: Language of subtitles (default: `en`).
*   `-j, --jobs`: Number of parallel download jobs (default: 8).
*   `--cookies-from-browser`: Browser to extract cookies from.

### `delete`: Delete a Channel

Remove a channel and all associated data from your database. **This action is irreversible.**

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID to delete (required).

### `export`: Export Transcripts

Export video transcripts for a specific channel.

```bash
# Export to TXT (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to VTT
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel name or ID to export transcripts for (required).
*   `-f, --format`: Export format: `txt`, `vtt` (default: `txt`).

### `search`: Full-Text Search

Perform keyword searches within saved transcripts. Use SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for advanced searches.

```bash
# Search in all channels
yt-fts search "[search query]"

# Search in a specific channel
yt-fts search "[search query]" --channel "[channel name or id]"

# Search in a specific video
yt-fts search "[search query]" --video-id "[video id]"

# Limit results
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# Export results to CSV
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a specific channel by name or ID.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit the number of results returned (default: 10).
*   `-e, --export`: Export search results to a CSV file.

**Advanced Search Syntax:**

*   **AND Search:** `yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"`
*   **OR Search:** `yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"`
*   **Wildcards:** `yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"`

### Semantic Search and RAG

Enable semantic search for more accurate and contextual searches. This requires an OpenAI or Gemini API key set in the environment variable `OPENAI_API_KEY` or `GEMINI_API_KEY`, or you can pass the key with the `--api-key` flag.

### `embeddings`: Generate Embeddings

Generate vector embeddings for semantic search, enabling context-aware results.

```bash
# Ensure your API key is set:
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"

# Specify the time interval (in seconds) to split text (default: 30).
# A larger interval increases accuracy but can lead to more text to read.
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID to generate embeddings for.
*   `--api-key`: API key (if not provided, it reads from `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable).
*   `-i, --interval`: Time interval (in seconds) to split transcripts into chunks (default: 30).

After the embeddings are saved, `(ss)` will appear next to the channel name when using the `list` command.

### `vsearch`: Semantic Search

Perform semantic (vector-based) searches, leveraging the generated embeddings.

```bash
# Search by channel name
yt-fts vsearch "[search query]" --channel "[channel name or id]"

# Search in a specific video
yt-fts vsearch "[search query]" --video-id "[video id]"

# Limit results
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# Export results to CSV
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a specific channel by name or ID.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit the number of results returned (default: 10).
*   `-e, --export`: Export search results to a CSV file.
*   `--api-key`: API key (if not provided, it reads from `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable).

### `llm`: Interactive Chat Bot

Engage in an interactive chat session using semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel name or ID to use (required).
*   `--api-key`: API key (if not provided, it reads from `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable).

### `summarize`: Summarize Video Transcripts

Get concise summaries of YouTube videos using LLMs.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use a different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use for summarization (e.g., "gpt-3.5-turbo").
*   `--api-key`: API key (if not provided, it reads from `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable).

**Example Output:**

```
In this video, 3Blue1Brown explores how large language models (LLMs) like GPT-3 might store facts within their vast...

 1 Introduction to Fact Storage in LLMs:
    • The video starts by questioning how LLMs store specific facts and
      introduces the idea that these facts might be stored in a particular part of the
      network known as multi-layer perceptrons (MLPs).
    • 0:00
 2 Overview of Transformers and MLPs:
    • Provides a refresher on transformers and explains that the video will focus
```

### `config`: View Configuration

Display current configuration settings, including database and ChromaDB paths.

```bash
yt-fts config
```

## How To

**Export Search Results:**

Both `search` and `vsearch` commands support exporting results to a CSV file using the `--export` flag:

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Remove a channel using the `delete` command:

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Update the subtitles for a channel:

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:**

Export a channel's transcripts:

```bash
# Export to VTT
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
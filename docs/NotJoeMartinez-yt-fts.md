# yt-fts: Your Command-Line YouTube Search Engine 🚀

**Quickly search and analyze YouTube transcripts with `yt-fts`, a powerful command-line tool that lets you find exactly what you're looking for across your favorite channels. Find the original repo [here](https://github.com/NotJoeMartinez/yt-fts).**

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

**Key Features:**

*   🔍 **Full-Text Search:**  Quickly search through YouTube transcripts using keywords and phrases.
*   🧠 **Semantic Search:**  Leverage OpenAI, Gemini, or ChromaDB to search for meaning, not just words.
*   💬 **Interactive LLM Chat Bot:**  Engage in conversations about video content using semantic search results for context.
*   📝 **Video Summarization:** Generate concise summaries of YouTube videos, complete with time-stamped links.
*   💾 **Database Driven:** Store and manage transcripts locally for fast and efficient searching.
*   🔄 **Easy Updates:** Keep your library fresh with automatic transcript updates.
*   📤 **Flexible Exporting:** Export search results and transcripts in various formats (CSV, TXT, VTT).

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Command-Line Usage

### `download`

Download subtitles for a YouTube channel or playlist.

```bash
# Download a channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"

# Download a playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist.
*   `-l, --language`: Specify the language of subtitles (default: `en`).
*   `-j, --jobs`: Set the number of parallel download jobs (default: 8, recommended: 4-16).
*   `--cookies-from-browser`: Use cookies from your browser (chrome, firefox, etc.) to avoid sign-in errors.

### `diagnose`

Diagnose and fix download issues.

```bash
yt-fts diagnose
```

**Options:**

*   `-u, --test-url`: Test a specific URL (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
*   `--cookies-from-browser`: Use cookies from your browser.
*   `-j, --jobs`: Set the number of parallel download jobs (default: 8).

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

*   `-t, --transcript`: Show the transcript for a video.
*   `-c, --channel`: Show the list of videos for a channel.
*   `-l, --library`: Show the list of channels in your library (same as default).

### `update`

Update subtitles for all channels or a specific channel.

```bash
# Update all channels
yt-fts update

# Update a specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Specify the channel to update.
*   `-l, --language`: Specify the language of subtitles (default: `en`).
*   `-j, --jobs`: Set the number of parallel download jobs (default: 8).
*   `--cookies-from-browser`: Use cookies from your browser.

### `delete`

Delete a channel and all its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`:  Specify the channel name or ID to delete (required).

### `export`

Export transcripts for a channel.

```bash
# Export to TXT (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to VTT
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Specify the channel to export (required).
*   `-f, --format`:  Choose the export format: `txt`, `vtt` (default: `txt`).

### `search` (Full Text Search)

Search for text within the saved transcripts.

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

*   `-c, --channel`: Specify the channel to search within.
*   `-v, --video-id`: Search within a specific video ID.
*   `-l, --limit`: Limit the number of search results (default: 10).
*   `-e, --export`: Export search results to a CSV file.

**Advanced Search Syntax:**

Utilize SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for advanced searching:

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR SEARCH
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# Wildcards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings` (Semantic Search Setup)

Enable semantic search for a channel using OpenAI or Gemini embeddings.

```bash
# Make sure OPENAI_API_KEY or GEMINI_API_KEY environment variable is set
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"

# Specify time interval (seconds) to split text (default: 30)
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Specify the channel to generate embeddings for.
*   `--api-key`: Provide your API key if not set as an environment variable.
*   `-i, --interval`: Set the interval (in seconds) to split transcripts into chunks (default: 30).

### `vsearch` (Semantic Search)

Perform a vector (semantic) search within channels that have embeddings enabled.

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

*   `-c, --channel`:  Specify the channel to search within.
*   `-v, --video-id`: Search within a specific video ID.
*   `-l, --limit`: Limit the number of search results (default: 10).
*   `-e, --export`: Export search results to a CSV file.
*   `--api-key`: Provide your API key.

### `llm` (Chat Bot)

Start an interactive chat session with a model using semantic search results for context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Specify the channel to use (required).
*   `--api-key`: Provide your API key.

### `summarize`

Summarize a YouTube video transcript.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use a different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`:  Specify the model to use for summarization.
*   `--api-key`:  Provide your API key.

### `config`

Show the current configuration settings, including database and ChromaDB paths.

```bash
yt-fts config
```

## How-To Guides

**Export Search Results:**

Export your search results to a CSV file:

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Use the `delete` command to remove a channel and all its associated data:

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

The `update` command is used to refresh the full-text search index for a channel.  *Note:  Currently, it does *not* update semantic search embeddings.*

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:**

Export all the transcripts of a channel:

```bash
# Export to VTT
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
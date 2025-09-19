# yt-fts: Unlock YouTube's Secrets with Powerful Full-Text Search

**Quickly search and analyze YouTube video transcripts with `yt-fts`, a command-line tool that indexes subtitles for lightning-fast keyword searches and advanced semantic analysis.** ([Original Repo](https://github.com/NotJoeMartinez/yt-fts))

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

## Key Features

*   **Full-Text Search:** Instantly search YouTube transcripts for specific keywords and phrases using SQLite's powerful search syntax.
*   **Semantic Search (with OpenAI/Gemini):** Leverage the power of semantic search to find videos based on meaning, not just exact words.
*   **LLM-Powered Chat Bot:** Engage in interactive conversations about YouTube content using the tool's semantic search results as context.
*   **Video Summarization:** Quickly generate summaries of YouTube videos with time-stamped links.
*   **Easy Installation:** Install via pip with a single command.
*   **Channel and Playlist Support:** Download and index subtitles from entire YouTube channels or playlists.
*   **Flexible Export Options:** Export search results and transcripts in various formats (CSV, VTT, TXT).
*   **Advanced Search Syntax:** Utilize SQLite's enhanced query syntax for precise and powerful searches including AND, OR, and wildcard searches.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

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
*   `-l, --language`:  Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16).
*   `--cookies-from-browser`: Browser to extract cookies from (e.g., `firefox`, `chrome`) to help with login errors.

### `diagnose`

Diagnose and fix common download issues (e.g., 403 errors).

```bash
yt-fts diagnose
```

**Options:**

*   `-u, --test-url`: URL to test (default:  `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
*   `--cookies-from-browser`: Browser to extract cookies from.
*   `-j, --jobs`: Number of parallel download jobs to test with.

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
*   `-c, --channel`: Show a list of videos for a channel.
*   `-l, --library`: Show a list of channels in the library.

### `update`

Update subtitles for all or a specific channel.

```bash
# Update all channels
yt-fts update

# Update a specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel name or ID to update.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel download jobs.
*   `--cookies-from-browser`: Browser to extract cookies from.

### `delete`

Delete a channel and all its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID to delete (required).

### `export`

Export transcripts for a channel.

```bash
# Export to TXT (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to VTT
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `-f, --format`: Export format: `txt` or `vtt` (default: `txt`).

### `search` (Full Text Search)

Perform full-text searches across saved transcripts.

```bash
# Search in all channels
yt-fts search "[search query]"

# Search in a specific channel
yt-fts search "[search query]" --channel "[channel name or id]"

# Limit results
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# Export results to CSV
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel name or ID to search within.
*   `-v, --video-id`: Video ID to search within.
*   `-l, --limit`: Number of results to return (default: 10).
*   `-e, --export`: Export search results to a CSV file.

**Advanced Search Syntax (SQLite Enhanced Query Syntax):**

Use operators for more complex searches:

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR SEARCH
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# Wildcards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### Semantic Search and RAG

Enable semantic search using the `embeddings` command. Requires an OpenAI or Gemini API key set as the `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable.

### `embeddings`

Generate embeddings for semantic search.

```bash
# Set API key (OpenAI)
# export OPENAI_API_KEY="[yourOpenAIKey]"

# Set API key (Gemini)
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"

# Adjust chunk interval (in seconds)
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID to generate embeddings for.
*   `--api-key`: API key (if not set, uses environment variables).
*   `-i, --interval`: Interval in seconds to split transcripts (default: 30).

After running this, the channel will have (ss) next to its name when listed, and you can use `vsearch`.

### `vsearch` (Semantic Search)

Perform semantic (vector) search based on meaning. Requires `embeddings` to be run first.

```bash
# Semantic search in a channel
yt-fts vsearch "[search query]" --channel "[channel name or id]"

# Search in a specific video
yt-fts vsearch "[search query]" --video-id "[video id]"

# Limit results
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"

# Export results to CSV
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Channel name or ID to search within.
*   `-v, --video-id`: Video ID to search within.
*   `-l, --limit`: Number of results to return (default: 10).
*   `-e, --export`: Export search results to a CSV file.
*   `--api-key`: API key (if not set, uses environment variables).

### `llm` (Chat Bot)

Start an interactive chat session with a model, leveraging semantic search for context. Requires semantic search to be enabled for the channel.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `--api-key`: API key (if not set, uses environment variables).

### `summarize`

Summarize YouTube video transcripts.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use a different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use for summarization.
*   `--api-key`: API key (if not set, uses environment variables).

## How To

**Export Search Results:**

Use the `--export` flag with both `search` and `vsearch` to save results to a CSV file in the current directory.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Use the `delete` command:

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Use the `update` command. Currently updates full-text search, *not* semantic search embeddings:

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:**

```bash
# Export to VTT
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
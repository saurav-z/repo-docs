# yt-fts: Full-Text Search for YouTube Subtitles

Unlock the power of YouTube's hidden knowledge with `yt-fts`, a command-line tool that lets you search and analyze video transcripts. [View the original repo](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

## Key Features

*   **Full-Text Search:** Quickly find specific keywords or phrases within YouTube video transcripts.
*   **Semantic Search:** Leverage AI to search based on the meaning of your queries using OpenAI, Gemini, or ChromaDB.
*   **LLM Chat Bot:** Interact with a chatbot that uses semantic search results as context, enabling in-depth conversations about video content.
*   **Video Summarization:** Generate concise summaries of YouTube videos, complete with timestamped links.
*   **Channel & Playlist Support:** Download and search subtitles from entire YouTube channels and playlists.
*   **Flexible Output:** Export search results to CSV, and transcripts to TXT or VTT formats.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`

Download subtitles for a YouTube channel or playlist and store them in a local database.

```bash
# Download a channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"

# Download a playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"

# Use cookies from browser
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
```

**Options:**

*   `-p, --playlist`: Download all videos from a playlist.
*   `-l, --language`: Specify the subtitle language (default: `en`).
*   `-j, --jobs`: Set the number of parallel download jobs (default: `8`, recommended: `4-16`).
*   `--cookies-from-browser`: Use cookies from your browser (e.g., `chrome`, `firefox`).

### `diagnose`

Diagnose and troubleshoot common YouTube download issues.

```bash
yt-fts diagnose
```

**Options:**

*   `-u, --test-url`: Test with a specific video URL (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
*   `--cookies-from-browser`: Browser to extract cookies from (e.g., `chrome`, `firefox`).
*   `-j, --jobs`: Number of parallel download jobs to test with (default: `8`).

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

*   `-t, --transcript`: Show the transcript for a specific video.
*   `-c, --channel`: Show a list of videos for a specific channel.
*   `-l, --library`: Show a list of all channels in your library.

### `update`

Update subtitles for existing channels in your library.

```bash
# Update all channels
yt-fts update

# Update a specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Specify the channel to update (by name or ID).
*   `-l, --language`: Specify the subtitle language (default: `en`).
*   `-j, --jobs`: Set the number of parallel download jobs (default: `8`).
*   `--cookies-from-browser`: Use cookies from your browser (e.g., `chrome`, `firefox`).

### `delete`

Delete a channel and its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Specify the channel to delete (required, by name or ID).

### `export`

Export transcripts for a channel to a specified format.

```bash
# Export transcripts to TXT format
yt-fts export --channel "3Blue1Brown" --format txt

# Export transcripts to VTT format
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Specify the channel to export transcripts for (required).
*   `-f, --format`: Specify the export format (options: `txt`, `vtt`; default: `txt`).

### `search` (Full Text Search)

Perform a full-text search across your saved transcripts.

```bash
# Search in all channels
yt-fts search "[search query]"

# Search in a specific channel
yt-fts search "[search query]" --channel "[channel name or id]"

# Limit search results
yt-fts search "[search query]" --limit 5 --channel "[channel name or id]"

# Export search results
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a specific channel (by name or ID).
*   `-v, --video-id`: Search within a specific video (by ID).
*   `-l, --limit`: Limit the number of search results returned (default: `10`).
*   `-e, --export`: Export search results to a CSV file.

**Advanced Search Syntax:**

Use SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for advanced search operators.

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR search
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# Wildcard search
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### Semantic Search and RAG

Enable semantic search for a channel to use AI for more intelligent searches.

### `embeddings`

Generate embeddings for a specified channel using OpenAI or Gemini.

```bash
# Generate embeddings (requires OPENAI_API_KEY or GEMINI_API_KEY environment variable)
# export OPENAI_API_KEY="[yourOpenAIKey]"
# or
# export GEMINI_API_KEY="[yourGeminiKey]"

yt-fts embeddings --channel "3Blue1Brown"

# Specify interval to split transcript into chunks
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Specify the channel to generate embeddings for (by name or ID).
*   `--api-key`: API key (if not provided, reads from `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable).
*   `-i, --interval`: Interval in seconds to split transcripts into chunks (default: `30`).

After the embeddings are created, you'll see `(ss)` next to the channel name when you list channels.

### `vsearch` (Semantic Search)

Perform a semantic (vector) search, which requires that you enable semantic search for a channel with `embeddings`. Results are sorted by similarity to the search query.

```bash
# Semantic search in a channel
yt-fts vsearch "[search query]" --channel "[channel name or id]"

# Limit search results
yt-fts vsearch "[search query]" --limit 5 --channel "[channel name or id]"

# Export search results
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a specific channel (by name or ID).
*   `-v, --video-id`: Search within a specific video (by ID).
*   `-l, --limit`: Limit the number of search results returned (default: `10`).
*   `-e, --export`: Export search results to a CSV file.
*   `--api-key`: API key (if not provided, reads from `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable).

### `llm` (Chat Bot)

Start an interactive chat session with a model, using semantic search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Specify the channel to use (required).
*   `--api-key`: API key (if not provided, reads from `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable).

### `summarize`

Generate a summary of a YouTube video transcript.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use a different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Specify the model to use for summarization.
*   `--api-key`: API key (if not provided, reads from `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable).

### `config`

Show the configuration settings, including database and Chroma paths.

```bash
yt-fts config
```

## Usage Examples

**Export Search Results:**

Export search results to a CSV file.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Delete a channel and all its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Update a channel's subtitles.

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:**

Export the entire transcript of a channel.

```bash
# Export to VTT
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
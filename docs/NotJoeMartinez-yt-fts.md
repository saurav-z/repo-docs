# yt-fts: Unleash the Power of YouTube Subtitle Search 

**yt-fts is a powerful command-line tool that lets you search YouTube video subtitles like never before.** ([Original Repo](https://github.com/NotJoeMartinez/yt-fts))

<p align="center">
  <img src="https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14" alt="yt-fts in action" width="600">
</p>

yt-fts downloads YouTube channel subtitles, stores them in a searchable SQLite database, and enables advanced search capabilities, including semantic search and LLM integrations.

**Key Features:**

*   **Full-Text Search:** Quickly find specific keywords or phrases within video transcripts.
*   **Semantic Search:** Utilize OpenAI or Gemini embeddings to find videos based on meaning and context, not just exact matches.
*   **LLM Integration:**  Chat with an LLM using the semantic search results as context for answers.
*   **Video Summarization:** Generate concise summaries of YouTube videos with time-stamped links.
*   **Channel and Playlist Support:** Download subtitles from entire channels or specific playlists.
*   **Flexible Export Options:** Export search results and transcripts in various formats (CSV, VTT, TXT).
*   **Easy Installation:** Simple installation via pip.

## Installation

Install yt-fts using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles

Download subtitles from a YouTube channel or playlist.

```bash
# Download channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"

# Download playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download from a playlist.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel downloads (default: 8).
*   `--cookies-from-browser`: Use browser cookies (chrome, firefox, etc.) to bypass sign-in requirements.

### `diagnose` - Diagnose Download Issues

Diagnose and troubleshoot download problems (e.g., 403 errors).

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test.
*   `--cookies-from-browser`: Use browser cookies.
*   `-j, --jobs`: Number of parallel jobs.

### `list` - List Saved Content

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

*   `-t, --transcript`: Show transcript for a video.
*   `-c, --channel`: Show videos for a channel.
*   `-l, --library`: Show channels in the library.

### `update` - Update Subtitles

Update subtitles for existing channels.

```bash
# Update all channels
yt-fts update

# Update specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel to update.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel jobs.
*   `--cookies-from-browser`: Use browser cookies.

### `delete` - Delete a Channel

Delete a channel and its associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).

### `export` - Export Transcripts

Export transcripts for a channel.

```bash
# Export to txt format (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to vtt format
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `-f, --format`: Export format (txt, vtt - default: txt).

### `search` - Full Text Search

Search for keywords or phrases in saved subtitles. Supports SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for advanced search.

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

*   `-c, --channel`: Channel name or ID.
*   `-v, --video-id`: Video ID.
*   `-l, --limit`: Number of results to return (default: 10).
*   `-e, --export`: Export results to CSV.

**Advanced Search Syntax:**

*   `AND`: Search for both terms.
*   `OR`: Search for either term.
*   `*`: Wildcard for partial word matching (e.g., `rea* kni*`).

### `embeddings` - Semantic Search Setup

Create embeddings for semantic search using OpenAI or Gemini. Requires an API key set as environment variable `OPENAI_API_KEY` or `GEMINI_API_KEY`, or via the `--api-key` flag.

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

*   `-c, --channel`: Channel name or ID.
*   `--api-key`: API key.
*   `-i, --interval`: Transcript chunking interval (default: 30 seconds).

### `vsearch` - Semantic (Vector) Search

Perform semantic search using embeddings (requires enabling semantic search for a channel with the `embeddings` command).

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

*   `-c, --channel`: Channel name or ID.
*   `-v, --video-id`: Video ID.
*   `-l, --limit`: Number of results to return (default: 10).
*   `-e, --export`: Export results to CSV.
*   `--api-key`: API key.

### `llm` - Interactive Chat with LLM

Chat with an LLM using semantic search results as context.  Requires semantic search to be enabled for the channel.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `--api-key`: API key.

### `summarize` - Video Summarization

Get a summary of a YouTube video transcript. Requires a valid YouTube video URL or ID.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: LLM Model to use.
*   `--api-key`: API key.

### `config` - Show Configuration

Display configuration settings, including database and Chroma paths.

```bash
yt-fts config
```

## How To

### Export Search Results

Export search results to a CSV file.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

### Delete a Channel

Delete a channel from the library.

```bash
yt-fts delete --channel "3Blue1Brown"
```

### Update a Channel

Update full text search data for a channel. (Semantic search updates not included).

```bash
yt-fts update --channel "3Blue1Brown"
```

### Export a Channel's Transcript

Export all transcripts for a channel.

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
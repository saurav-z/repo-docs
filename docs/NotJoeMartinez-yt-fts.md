# yt-fts: Unleash the Power of Full Text Search for YouTube!

Quickly search YouTube video subtitles and unlock hidden insights with yt-fts, a powerful command-line tool. [See the original repo](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full Text Search:**  Rapidly search within YouTube video transcripts for keywords and phrases.
*   **Semantic Search:** Leverage AI (OpenAI/Gemini/ChromaDB) for context-aware searches using the `vsearch` command.
*   **AI-Powered Chat Bot:**  Engage in conversational exploration of video content using the `llm` command.
*   **Video Summarization:**  Get concise summaries of YouTube videos with time-stamped links, using the `summarize` command.
*   **Easy Installation:** Simple installation via pip.
*   **Channel & Playlist Support:** Download subtitles from entire channels or playlists.
*   **Flexible Export Options:**  Export search results and transcripts in various formats (CSV, TXT, VTT).
*   **Robust Download Management:** Handle errors with diagnose command, parallel downloads, and browser cookie integration.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### Download Subtitles (`download`)

Download subtitles for a YouTube channel or playlist. Use the `--cookies-from-browser` flag to use cookies from your browser to avoid sign-in issues. The `update` command can be run repeatedly to gradually add videos to the database.

```bash
# Download a channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"

# Download a playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download videos from a playlist.
*   `-l, --language`: Specify subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel download jobs (default: `8`, recommended: `4-16`).
*   `--cookies-from-browser`: Browser to extract cookies from (e.g., `chrome`, `firefox`).

### Diagnose Download Issues (`diagnose`)

Diagnose and troubleshoot common YouTube download problems.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test (default: `https://www.youtube.com/watch?v=dQw4w9WgXcQ`).
*   `--cookies-from-browser`: Browser for cookie extraction.
*   `-j, --jobs`: Number of parallel jobs (default: 8).

### List Data (`list`)

View saved channels, videos, and transcripts.

```bash
# List all channels
yt-fts list

# List videos for a specific channel
yt-fts list --channel "3Blue1Brown"

# Show a video transcript
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**

*   `-t, --transcript`: Show a video's transcript.
*   `-c, --channel`: Show videos for a channel.
*   `-l, --library`: Show a list of channels in your library.

### Update Subtitles (`update`)

Update subtitles for existing channels in your library.

```bash
# Update all channels
yt-fts update

# Update a specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Update a specific channel (name or ID).
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Parallel jobs (default: 8).
*   `--cookies-from-browser`: Browser for cookie extraction.

### Delete Channel (`delete`)

Delete a channel and all associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`:  Channel name or ID to delete (required).

### Export Transcripts (`export`)

Export transcripts for a channel to a directory in the current directory.

```bash
# Export to TXT (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to VTT
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `-f, --format`: Export format (default: `txt`, options: `txt`, `vtt`).

### Full Text Search (`search`)

Search for keywords and phrases within downloaded subtitles.

```bash
# Search all channels
yt-fts search "[your search query]"

# Search a specific channel
yt-fts search "[your search query]" --channel "[channel name or id]"

# Search a specific video
yt-fts search "[your search query]" --video-id "[video id]"

# Limit results
yt-fts search "[your search query]" --limit "[number of results]" --channel "[channel name or id]"

# Export results to CSV
yt-fts search "[your search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Number of results to return (default: 10).
*   `-e, --export`: Export results to a CSV file.

**Advanced Search Syntax (SQLite Enhanced Query Syntax):**

*   **AND:** `yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"`
*   **OR:** `yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"`
*   **Wildcards:** `yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"`

### Semantic Search with Embeddings and RAG

Enable semantic search for a channel using OpenAI or Gemini API keys via embeddings.  Requires an OpenAI or Gemini API key, set as an environment variable `OPENAI_API_KEY` or `GEMINI_API_KEY`.

```bash
# Set your API key (e.g., in your .bashrc or .zshrc)
# export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
# or
# export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"

yt-fts embeddings --channel "3Blue1Brown"

# Specify chunk interval (default: 30 seconds).
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`:  Channel name or ID.
*   `--api-key`:  API key (overrides environment variables).
*   `-i, --interval`:  Interval (in seconds) for splitting transcripts (default: 30).

### Semantic Vector Search (`vsearch`)

Perform semantic searches based on your query, requires the channel to have embeddings.

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

*   `-c, --channel`: Channel name or ID.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Results limit (default: 10).
*   `-e, --export`: Export results to a CSV file.
*   `--api-key`: API key (overrides environment variables).

### AI Chat Bot (`llm`)

Start an interactive chat session using the semantic search results as context. The channel must have semantic search enabled.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `--api-key`: API Key (overrides environment variables).

### Summarize Videos (`summarize`)

Get summaries of YouTube video transcripts.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use a different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use (e.g., `gpt-3.5-turbo`).
*   `--api-key`: API key (overrides environment variables).

### Config (`config`)

Show configuration settings including the database and chroma paths.

```bash
yt-fts config
```

## How To

**Export Search Results:**

Export search results to a CSV file:

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Remove a channel and its data:

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Update subtitles for a channel:

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:**

Export a channel's transcript to a file:

```bash
# Export to VTT
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
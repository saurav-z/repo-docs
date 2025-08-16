# yt-fts: Supercharge Your YouTube Research with Full Text Search

**Quickly and efficiently search and analyze YouTube video transcripts using powerful command-line tools. Find the exact moments you need with advanced search, semantic understanding, and AI-powered features.** ([View on GitHub](https://github.com/NotJoeMartinez/yt-fts))

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

## Key Features:

*   **Full Text Search:** Search within YouTube transcripts using keywords, phrases, and advanced search syntax.
*   **Semantic Search:** Leverage AI-powered semantic search to find videos based on meaning and context. Integrates with OpenAI, Gemini, and ChromaDB for intelligent results.
*   **LLM-Powered Chatbot:** Converse with a chatbot that uses semantic search results for context, enabling you to ask complex questions about YouTube content.
*   **Video Summarization:** Quickly get summaries of YouTube videos with time-stamped links.
*   **Channel & Playlist Support:** Download and search transcripts from entire YouTube channels or playlists.
*   **Flexible Exporting:** Export search results and transcripts in various formats (CSV, TXT, VTT).

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`: Download Subtitles

Download subtitles from a YouTube channel or playlist and store them in a searchable database.

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

### `diagnose`: Troubleshoot Download Issues

Diagnose and resolve issues related to downloading subtitles, such as 403 errors.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**
*   `-u, --test-url`: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list`: View Library Contents

List saved channels, videos, and transcripts within the database.

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

### `update`: Refresh Subtitles

Update subtitles for all channels or a specific channel in your library.

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

### `delete`: Remove Channels

Delete a channel and all its associated data from your library.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**
*   `-c, --channel`: The name or id of the channel to delete (required)

### `export`: Export Transcripts

Export transcripts for a specific channel in various formats.

```bash
# Export to txt format (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to vtt format
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**
*   `-c, --channel`: The name or id of the channel to export transcripts for (required)
*   `-f, --format`: The format to export transcripts to. Supported formats: txt, vtt (default: txt)

### `search`: Full Text Search

Perform a full text search within the downloaded transcripts. Uses sqlite [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries).

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

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR SEARCH
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings`: Enable Semantic Search

Generate embeddings for a channel to enable semantic search and LLM-powered features. Requires an OpenAI or Gemini API key set in the environment variable `OPENAI_API_KEY` or `GEMINI_API_KEY`, or pass the key with the `--api-key` flag.

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

### `vsearch`: Semantic (Vector) Search

Search YouTube content using semantic similarity. Requires that you enable semantic search for a channel with `embeddings`.

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

### `llm`: Conversational Chatbot

Engage in an interactive chat session using semantic search results as context. The channel must have semantic search enabled.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**
*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize`: Summarize Videos

Summarize YouTube video transcripts. Requires a valid YouTube video URL or video ID as an argument.

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

### `config`: View Configuration

Show configuration settings, including database and Chroma paths.

```bash
yt-fts config
```

## How To

**Export search results:**

For both the `search` and `vsearch` commands you can export the results to a csv file with
the `--export` flag. and it will save the results to a csv file in the current directory.
```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a channel:**
You can delete a channel with the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a channel:**
The update command currently only works for full text search and will not update the
semantic search embeddings.

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export all of a channel's transcript:**

This command will create a directory in current working directory with the YouTube
channel id of the specified channel.
```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
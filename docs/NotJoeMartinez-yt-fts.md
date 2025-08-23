# yt-fts: Unleash the Power of YouTube Search 🚀

Quickly and easily search the full transcripts of YouTube channels with `yt-fts`, a powerful command-line tool. ([Original Repo](https://github.com/NotJoeMartinez/yt-fts))

![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)

## Key Features

*   **Full Text Search:** Search YouTube channel transcripts using keywords and phrases.
*   **Semantic Search:** Leverage OpenAI and Gemini embeddings for more intelligent search results.
*   **LLM Integration:** Chat with channels using their transcripts as context.
*   **Video Summarization:** Generate concise summaries of YouTube videos.
*   **Flexible Data Management:** Download, update, list, and delete channels and transcripts.
*   **Export Results:** Export search results to CSV for easy analysis.
*   **Advanced Search Syntax:** Utilize SQLite's enhanced query syntax for precise results.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`: Download Subtitles

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

### `diagnose`: Diagnose Download Issues

Diagnose 403 errors and other download issues.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: URL to test with (default: https://www.youtube.com/watch?v=dQw4w9WgXcQ)
*   `--cookies-from-browser`: Browser to extract cookies from
*   `-j, --jobs`: Number of parallel download jobs to test with (default: 8)

### `list`: List Saved Channels, Videos, and Transcripts

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

### `update`: Update Subtitles

Update subtitles for all or a specific channel.

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

### `delete`: Delete a Channel

Delete a channel and all its data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to delete (required)

### `export`: Export Transcripts

Export transcripts for a channel.

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

Perform full-text searches within saved channel transcripts.

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

Use SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for more powerful searches.

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR SEARCH
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings`: Semantic Search

Enable semantic search for a channel using OpenAI or Gemini embeddings. Requires an API key set in `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable, or use the `--api-key` flag.

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

After generating embeddings, you can use the `vsearch` and `llm` commands.

### `vsearch`: Semantic Search

Perform semantic (vector-based) searches using embeddings.

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

### `llm`: Chat Bot

Start an interactive chat session with a model, using semantic search results as context. The channel must have semantic search enabled.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize`: Summarize a Video

Summarize a YouTube video transcript. Requires a valid YouTube video URL or video ID.

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

### `config`: Show Configuration

Show config settings, including database and Chroma paths.

```bash
yt-fts config
```

## How To

**Export Search Results:**

Export search results to a CSV file using the `--export` flag with the `search` and `vsearch` commands.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:**

Delete a channel with the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:**

Update a channel using the `update` command. *Note: The `update` command currently only updates full-text search and does not update semantic search embeddings.*

```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:**

Export a channel's transcripts with the following command, which creates a directory with the channel ID.

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
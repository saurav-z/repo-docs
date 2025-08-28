# yt-fts: Supercharge Your YouTube Research with Full-Text Search

**Quickly search, analyze, and extract insights from YouTube channel transcripts with `yt-fts` - your command-line companion for in-depth video exploration.** ([View the original repo](https://github.com/NotJoeMartinez/yt-fts))

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

`yt-fts` is a powerful command-line tool that leverages `yt-dlp` to download YouTube channel subtitles, stores them in a searchable SQLite database, and offers advanced features like semantic search and AI-powered summarization.

**Key Features:**

*   **Full-Text Search:** Quickly find specific keywords or phrases across all your downloaded transcripts.
*   **Semantic Search:** Leverage the power of OpenAI and Gemini embeddings for more intelligent search results.
*   **AI-Powered Summarization:** Generate concise summaries of YouTube videos with time-stamped links.
*   **Flexible Download Options:** Download subtitles for channels, playlists, and individual videos.
*   **Advanced Search Syntax:** Utilize SQLite's Enhanced Query Syntax for complex searches, including AND, OR, and wildcard searches.
*   **LLM Chat Bot:** Interact with a chat bot, with context from your search results.
*   **Easy Installation:** Simple installation via pip.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download`: Download Subtitles

Download subtitles for a YouTube channel or playlist. Use `--cookies-from-browser` to handle potential sign-in issues.

```bash
# Download channel
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"

# Download playlist
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download playlist content.
*   `-l, --language`:  Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel download jobs (default: 8, recommended: 4-16).
*   `--cookies-from-browser`: Use cookies from your browser (chrome, firefox, etc.).

### `diagnose`: Troubleshoot Download Issues

Diagnose and resolve common download problems.

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: Test URL.
*   `--cookies-from-browser`: Browser to use for cookies.
*   `-j, --jobs`: Number of parallel download jobs to test.

### `list`: List Saved Data

List your saved channels, videos, and transcripts.

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

*   `-t, --transcript`: Show video transcript.
*   `-c, --channel`: Show videos for a channel.
*   `-l, --library`: Show a list of channels in the library.

### `update`: Update Subtitles

Update subtitles for all or specific channels.

```bash
# Update all channels
yt-fts update

# Update specific channel
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: Channel name or ID to update.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Number of parallel download jobs.
*   `--cookies-from-browser`: Browser for cookies.

### `delete`: Delete Channel Data

Delete a channel and its associated data. Requires confirmation.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).

### `export`: Export Transcripts

Export transcripts in various formats.

```bash
# Export to txt format (default)
yt-fts export --channel "3Blue1Brown" --format txt

# Export to vtt format
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `-f, --format`: Export format (`txt`, `vtt`). (default: txt)

### `search`: Full-Text Search

Search for keywords within downloaded transcripts.

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

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Number of results to display (default: 10).
*   `-e, --export`: Export search results to CSV.

**Advanced Search Syntax:** Utilize SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for powerful searches.

```bash
# AND search
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"

# OR SEARCH
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"

# wild cards
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings`: Semantic Search Setup

Enable semantic search for a channel using OpenAI or Gemini embeddings.  Requires setting the `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable, or using `--api-key`.

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
*   `--api-key`: API key (if not provided, reads from environment variables).
*   `-i, --interval`: Chunking interval in seconds (default: 30).

After running this command, the channel will be marked with `(ss)` when listed, and you can use `vsearch`.

### `vsearch`: Semantic Search

Perform vector-based semantic search on channels with embeddings enabled.

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
*   `-l, --limit`: Result limit (default: 10).
*   `-e, --export`: Export results to CSV.
*   `--api-key`: API key (if not provided, reads from environment variables).

### `llm`: Chat Bot

Start an interactive chatbot session, using search results as context.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `--api-key`: API key (if not provided, reads from environment variables).

### `summarize`: Video Summarization

Generate summaries for YouTube videos.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
# or
yt-fts summarize "9-Jl0dxWQs8"

# Use different model
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: AI model to use.
*   `--api-key`: API key (if not provided, reads from environment variables).

### `config`: View Configuration

Display current configuration settings.

```bash
yt-fts config
```

## How To Guides

**Exporting Search Results**

Export search results from both `search` and `vsearch` to a CSV file.

```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Deleting a Channel**

Delete a channel and all associated data using the `delete` command.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Updating a Channel's Subtitles**

Update a channel's subtitles using the `update` command.

```bash
yt-fts update --channel "3Blue1Brown"
```

**Exporting a Channel's Transcript**

Export a channel's transcript using the `export` command.

```bash
# Export to vtt
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```
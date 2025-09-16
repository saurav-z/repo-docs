# yt-fts: Unleash the Power of YouTube Transcript Search!

**yt-fts** is a command-line tool that allows you to fully index and search YouTube video transcripts, enabling powerful keyword and semantic search capabilities. Access the original repo [here](https://github.com/NotJoeMartinez/yt-fts).

[![yt-fts Demo](https://github.com/NotJoeMartinez/yt-fts/assets/39905973/6ffd8962-d060-490f-9e73-9ab179402f14)](https://github.com/NotJoeMartinez/yt-fts)

**Key Features:**

*   **Full-Text Search:** Quickly find videos containing specific keywords or phrases within channel transcripts.
*   **Semantic Search:** Leverage AI to search based on meaning using OpenAI or Gemini embeddings.
*   **Channel and Playlist Support:** Download and search transcripts from entire YouTube channels or playlists.
*   **Time-Stamped Results:** Get instant access to the precise timestamps within videos where your search terms appear.
*   **LLM-Powered Chatbot:** Engage in interactive conversations with your data using the LLM chat bot to learn from video transcripts.
*   **Video Summarization:** Generate concise summaries of YouTube videos, with time-stamped links.
*   **Advanced Search:** Supports SQLite's Enhanced Query Syntax, including AND/OR, wildcards, and prefix queries.
*   **Flexible Export:** Export search results to CSV and channel transcripts to TXT/VTT formats.
*   **Easy Installation:** Install quickly with `pip install yt-fts`.

## Installation

Install `yt-fts` using pip:

```bash
pip install yt-fts
```

## Commands

### `download` - Download Subtitles

Downloads subtitles for a specified channel or playlist. Use `--cookies-from-browser` to resolve potential sign-in issues.

```bash
yt-fts download --jobs 5 "https://www.youtube.com/@3blue1brown"
yt-fts download --cookies-from-browser firefox "https://www.youtube.com/@3blue1brown"
yt-fts download --playlist "https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab"
```

**Options:**

*   `-p, --playlist`: Download from a playlist.
*   `-l, --language`: Subtitle language (default: `en`).
*   `-j, --jobs`: Parallel download jobs (default: 8).
*   `--cookies-from-browser`: Use browser cookies (chrome, firefox, etc.).

### `diagnose` - Troubleshoot Download Issues

Tests your connection to YouTube and provides solutions for common download errors (e.g., 403 errors).

```bash
yt-fts diagnose
yt-fts diagnose --test-url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" --cookies-from-browser firefox
```

**Options:**

*   `-u, --test-url`: Test URL.
*   `--cookies-from-browser`: Use browser cookies.
*   `-j, --jobs`: Parallel download jobs.

### `list` - View Data

Lists saved channels, videos, and transcripts.

```bash
yt-fts list
yt-fts list --channel "3Blue1Brown"
yt-fts list --transcript "dQw4w9WgXcQ"
```

**Options:**

*   `-t, --transcript`: Show a video's transcript.
*   `-c, --channel`: Show videos for a channel.
*   `-l, --library`: Show the list of channels.

### `update` - Refresh Subtitles

Updates subtitles for existing channels.

```bash
yt-fts update
yt-fts update --channel "3Blue1Brown" --jobs 5
```

**Options:**

*   `-c, --channel`: The name or id of the channel to update
*   `-l, --language`: Language of the subtitles to download (default: en)
*   `-j, --jobs`: Number of parallel download jobs (default: 8)
*   `--cookies-from-browser`: Browser to extract cookies from

### `delete` - Remove Channels

Deletes a channel and all associated data.

```bash
yt-fts delete --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to delete (required).

### `export` - Export Transcripts

Exports transcripts to text or VTT format.

```bash
yt-fts export --channel "3Blue1Brown" --format txt
yt-fts export --channel "3Blue1Brown" --format vtt
```

**Options:**

*   `-c, --channel`: Channel name or ID (required).
*   `-f, --format`: Export format (txt, vtt; default: txt).

### `search` - Full-Text Search

Perform full-text searches within saved transcripts.

```bash
yt-fts search "[search query]"
yt-fts search "[search query]" --channel "[channel name or id]"
yt-fts search "[search query]" --video-id "[video id]"
yt-fts search "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts search "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit the number of results (default: 10).
*   `-e, --export`: Export results to CSV.

**Advanced Search Syntax:** Use SQLite's [Enhanced Query Syntax](https://www.sqlite.org/fts3.html#full_text_index_queries) for more complex searches, including AND/OR operators and wildcards.

```bash
yt-fts search "knife AND Malibu" --channel "The Tim Dillon Show"
yt-fts search "knife OR Malibu" --channel "The Tim Dillon Show"
yt-fts search "rea* kni* Mali*" --channel "The Tim Dillon Show"
```

### `embeddings` - Enable Semantic Search

Generate embeddings using OpenAI or Gemini API keys to enable semantic search. Requires the `OPENAI_API_KEY` or `GEMINI_API_KEY` environment variable or the `--api-key` flag.

```bash
yt-fts embeddings --channel "3Blue1Brown"
yt-fts embeddings --interval 60 --channel "3Blue1Brown"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to generate embeddings for
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)
*   `-i, --interval`: Interval in seconds to split the transcripts into chunks (default: 30)

### `vsearch` - Semantic Search

Perform semantic searches, returning results based on relevance. Requires semantic search enabled for the channel (using `embeddings`).

```bash
yt-fts vsearch "[search query]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --video-id "[video id]"
yt-fts vsearch "[search query]" --limit "[number of results]" --channel "[channel name or id]"
yt-fts vsearch "[search query]" --export --channel "[channel name or id]"
```

**Options:**

*   `-c, --channel`: Search within a specific channel.
*   `-v, --video-id`: Search within a specific video.
*   `-l, --limit`: Limit the number of results (default: 10).
*   `-e, --export`: Export results to CSV.
*   `--api-key`: API key

### `llm` - Chat Bot

Start an interactive chat session with a LLM. Requires a channel with semantic search enabled.

```bash
yt-fts llm --channel "3Blue1Brown" "How does back propagation work?"
```

**Options:**

*   `-c, --channel`: The name or id of the channel to use (required)
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `summarize` - Video Summarization

Generates a summary of a YouTube video's transcript, with time-stamped links.

```bash
yt-fts summarize "https://www.youtube.com/watch?v=9-Jl0dxWQs8"
yt-fts summarize "9-Jl0dxWQs8"
yt-fts summarize --model "gpt-3.5-turbo" "9-Jl0dxWQs8"
```

**Options:**

*   `--model, -m`: Model to use in summary
*   `--api-key`: API key (if not provided, reads from OPENAI_API_KEY or GEMINI_API_KEY environment variable)

### `config` - View Configuration

Displays the current configuration settings, including database and Chroma paths.

```bash
yt-fts config
```

## How To

**Export Search Results:** Export the output of `search` and `vsearch` to CSV files.
```bash
yt-fts search "life in the big city" --export
yt-fts vsearch "existing in large metropolaten center" --export
```

**Delete a Channel:** Use the `delete` command.
```bash
yt-fts delete --channel "3Blue1Brown"
```

**Update a Channel:** Use the `update` command.
```bash
yt-fts update --channel "3Blue1Brown"
```

**Export a Channel's Transcript:** Export channel transcripts to TXT or VTT formats.
```bash
yt-fts export --channel "[id/name]" --format "[vtt/txt]"
```